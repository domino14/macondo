package wmp

import (
	"errors"
	"testing"

	wglconfig "github.com/domino14/word-golib/config"
)

const mb = 1 << 20

func TestPolicySizeLimit(t *testing.T) {
	// What the Lambda sets: read what's seeded and small enough, never build.
	lambda := Policy{Enabled: true, BuildIfMissing: false, MaxBytes: 200 * mb}
	for _, tc := range []struct {
		name    string
		size    int64
		allowed bool
	}{
		{"ECWL", 45 * mb, true},
		{"NWL23", 122 * mb, true},
		{"CSW24", 170 * mb, true},
		{"FRA24", 202 * mb, false},
		{"RD29", 416 * mb, false},
		{"SLV26", 746 * mb, false},
	} {
		if got := lambda.allowsSize(tc.size); got != tc.allowed {
			t.Errorf("%s (%d MB): allowsSize = %v, want %v", tc.name, tc.size/mb, got, tc.allowed)
		}
	}

	// The default: no limit at all.
	if !(Policy{Enabled: true}).allowsSize(746 * mb) {
		t.Error("a zero MaxBytes should mean no limit")
	}
}

func TestPolicyDefaults(t *testing.T) {
	// Nothing set: build one that's missing, read one that isn't, no limit.
	for _, k := range []string{EnvEnabled, EnvBuildIfMissing, EnvMaxMB} {
		t.Setenv(k, "")
	}
	p := policyFromEnv()
	if !p.Enabled || !p.BuildIfMissing || p.MaxBytes != 0 {
		t.Errorf("defaults should be enabled, build-if-missing, unlimited; got %+v", p)
	}
}

func TestPolicyFromEnv(t *testing.T) {
	t.Setenv(EnvBuildIfMissing, "false")
	t.Setenv(EnvMaxMB, "200")
	p := policyFromEnv()
	if p.BuildIfMissing || p.MaxBytes != 200*mb || !p.Enabled {
		t.Errorf("got %+v", p)
	}

	t.Setenv(EnvEnabled, "false")
	if policyFromEnv().Enabled {
		t.Error("MACONDO_WMP_ENABLED=false should disable word maps")
	}

	// Nonsense falls back to the default rather than turning something off.
	t.Setenv(EnvEnabled, "yes-please")
	t.Setenv(EnvMaxMB, "two hundred")
	p = policyFromEnv()
	if !p.Enabled || p.MaxBytes != 0 {
		t.Errorf("bad values should fall back to defaults; got %+v", p)
	}
}

// With building turned off, a lexicon whose word map isn't on disk must fall
// back to the KWG move generator rather than paying for a build -- the whole
// point of the setting on a platform that pays per cold start.
func TestEnsureWMPRespectsBuildIfMissing(t *testing.T) {
	defer SetPolicy(policyFromEnv())
	cfg := &wglconfig.Config{DataPath: t.TempDir()}

	SetPolicy(Policy{Enabled: true, BuildIfMissing: false})
	if _, err := EnsureWMP(cfg, "NWL23"); !errors.Is(err, ErrDisabled) {
		t.Errorf("BuildIfMissing=false should refuse to build; got %v", err)
	}

	SetPolicy(Policy{Enabled: false, BuildIfMissing: true})
	if _, err := EnsureWMP(cfg, "NWL23"); !errors.Is(err, ErrDisabled) {
		t.Errorf("Enabled=false should refuse outright; got %v", err)
	}

	// With building on, it gets as far as needing the KWG -- a different
	// failure, which is how we know it tried.
	SetPolicy(Policy{Enabled: true, BuildIfMissing: true})
	if _, err := EnsureWMP(cfg, "NWL23"); err == nil || errors.Is(err, ErrDisabled) {
		t.Errorf("BuildIfMissing=true should attempt a build; got %v", err)
	}
}
