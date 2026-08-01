package wmp

import (
	"errors"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/rs/zerolog/log"
)

// A word map buys move generation speed for memory and startup time, and that
// trade lands differently in different places. Three environment variables
// decide it, each answering one question:
//
//	MACONDO_WMP_ENABLED            use word maps at all?          (default true)
//	MACONDO_WMP_BUILD_IF_MISSING   build one that isn't on disk?  (default true)
//	MACONDO_WMP_MAX_MB             how big may one be?            (default 0, no limit)
//
// The defaults are what a developer wants: build a word map the first time a
// lexicon needs one, read it from disk every time after, no size limit.
//
// A Lambda wants the opposite of the build: it pays for one on every cold
// start, and building CSW24's takes about as long as reading it (1.4s either
// way), while SLV26's takes 10.4s to build or 8.6s to read against a KWG move
// generator that costs neither. So it sets
//
//	MACONDO_WMP_BUILD_IF_MISSING=false
//	MACONDO_WMP_MAX_MB=200
//
// and gets: read whatever is seeded on the volume and small enough, and fall
// back to the KWG move generator for anything missing or larger.

const (
	EnvEnabled        = "MACONDO_WMP_ENABLED"
	EnvBuildIfMissing = "MACONDO_WMP_BUILD_IF_MISSING"
	EnvMaxMB          = "MACONDO_WMP_MAX_MB"
)

// ErrDisabled is returned when policy says not to use a word map. Callers
// treat it like a missing one: fall back to the KWG move generator.
var ErrDisabled = errors.New("word map disabled by policy")

// Policy is the resolved word map policy for this process.
type Policy struct {
	// Enabled false means never use a word map, whatever is on disk.
	Enabled bool
	// BuildIfMissing false means only ever use a word map that is already on
	// disk. A lexicon without one falls back to the KWG move generator rather
	// than paying the build.
	BuildIfMissing bool
	// MaxBytes is the largest word map that will be used; a bigger one is
	// skipped in favour of the KWG move generator. Zero means no limit.
	//
	// It can only be applied to a word map already on disk, since the size of
	// one is not predictable before it is built -- the ratio to the KWG it
	// comes from runs from 27x to 94x across lexica.
	MaxBytes int64
}

var (
	policyOnce sync.Once
	policy     Policy
	policyMu   sync.RWMutex
)

// CurrentPolicy returns the policy in force, reading it from the environment
// the first time it is asked for and logging what it resolved to.
func CurrentPolicy() Policy {
	policyOnce.Do(func() {
		p := policyFromEnv()
		policyMu.Lock()
		policy = p
		policyMu.Unlock()
		log.Info().Bool("enabled", p.Enabled).
			Bool("build_if_missing", p.BuildIfMissing).
			Int64("max_mb", p.MaxBytes/(1<<20)).
			Msg("wmp-policy")
	})
	policyMu.RLock()
	defer policyMu.RUnlock()
	return policy
}

// SetPolicy overrides the policy for this process, for a caller that
// configures itself by some means other than the environment.
func SetPolicy(p Policy) {
	CurrentPolicy() // so the once-only env read can't clobber this later
	policyMu.Lock()
	policy = p
	policyMu.Unlock()
}

func policyFromEnv() Policy {
	return Policy{
		Enabled:        boolFromEnv(EnvEnabled, true),
		BuildIfMissing: boolFromEnv(EnvBuildIfMissing, true),
		MaxBytes:       mbFromEnv(EnvMaxMB, 0),
	}
}

func boolFromEnv(key string, def bool) bool {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		log.Warn().Str(key, v).Bool("using", def).
			Msg("word map setting is not a boolean")
		return def
	}
	return b
}

func mbFromEnv(key string, def int64) int64 {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def << 20
	}
	mb, err := strconv.ParseInt(v, 10, 64)
	if err != nil || mb < 0 {
		log.Warn().Str(key, v).Int64("using_mb", def).
			Msg("word map size limit is not a non-negative number of megabytes")
		return def << 20
	}
	return mb << 20
}

// allowsSize reports whether a word map of the given size may be used. A zero
// MaxBytes -- the default, and what an unset MACONDO_WMP_MAX_MB produces --
// means there is no limit.
func (p Policy) allowsSize(size int64) bool {
	return p.MaxBytes == 0 || size <= p.MaxBytes
}
