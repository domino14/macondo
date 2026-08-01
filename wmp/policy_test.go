package wmp

import "testing"

func TestPolicySizeRules(t *testing.T) {
	const mb = 1 << 20
	auto := Policy{Mode: ModeAuto, MaxBytes: 400 * mb, MaxReadBytes: 200 * mb}

	for _, tc := range []struct {
		name              string
		size              int64
		wantUse, wantRead bool
	}{
		// The sizes that prompted this: CSW24 is read, RD29 is over the read
		// limit so it is rebuilt, SLV26 is over the limit entirely.
		{"CSW24", 170 * mb, true, true},
		{"DISC2", 232 * mb, true, false},
		{"RD29", 416 * mb, false, false},
		{"SLV26", 746 * mb, false, false},
	} {
		if got := auto.allowsSize(tc.size); got != tc.wantUse {
			t.Errorf("%s (%d MB): allowsSize = %v, want %v", tc.name, tc.size/mb, got, tc.wantUse)
		}
		if tc.wantUse {
			if got := !auto.prefersRebuild(tc.size); got != tc.wantRead {
				t.Errorf("%s (%d MB): reads from disk = %v, want %v", tc.name, tc.size/mb, got, tc.wantRead)
			}
		}
	}

	// always ignores both limits; never refuses outright.
	always := Policy{Mode: ModeAlways, MaxBytes: 1 * mb, MaxReadBytes: 1 * mb}
	if !always.allowsSize(746*mb) || always.prefersRebuild(746*mb) {
		t.Error("ModeAlways should use any word map, read from disk")
	}
	if (Policy{Mode: ModeNever}).useWordMap() {
		t.Error("ModeNever should not use a word map")
	}
	// A zero limit means no limit.
	noLimit := Policy{Mode: ModeAuto}
	if !noLimit.allowsSize(1<<40) || noLimit.prefersRebuild(1<<40) {
		t.Error("zero limits should mean unlimited")
	}
}

func TestPolicyFromEnv(t *testing.T) {
	t.Setenv(EnvMode, "always")
	t.Setenv(EnvMaxMB, "123")
	if p := policyFromEnv(); p.Mode != ModeAlways || p.MaxBytes != 123<<20 {
		t.Errorf("got %+v", p)
	}
	t.Setenv(EnvMode, "nonsense")
	t.Setenv(EnvMaxMB, "not-a-number")
	if p := policyFromEnv(); p.Mode != ModeAuto || p.MaxBytes != defaultMaxMB<<20 {
		t.Errorf("bad values should fall back to defaults; got %+v", p)
	}
}
