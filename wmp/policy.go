package wmp

import (
	"errors"
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/rs/zerolog/log"
)

// A word map is a speed/size tradeoff that doesn't pay off the same way
// everywhere. On a long-lived server with memory to spare it is always worth
// having. On a Lambda that rebuilds or re-reads it on every cold start, a
// large one costs more in startup latency than the move generator it replaces
// saves: SLV26's is 746 MB, which is 8.6s to read off EFS or 10.4s to build,
// against a KWG move generator that needs neither.
//
// The policy below decides, per lexicon, whether to use a word map at all and
// whether to read it or rebuild it. It is read from the environment so a
// deployment can set it without a code change:
//
//	MACONDO_WMP_MODE          auto (default) | always | never
//	MACONDO_WMP_MAX_MB        auto only; a .wmp on disk larger than this is
//	                          not used at all (default 400)
//	MACONDO_WMP_MAX_READ_MB   auto only; a .wmp on disk larger than this is
//	                          rebuilt in memory rather than read from disk
//	                          (default 0, meaning always read it)
//
// "always" ignores both sizes: use whatever word map exists or can be built.
// That is the setting for an always-on server with a big heap. "never" skips
// word maps entirely and leaves every caller on the KWG move generator.

type Mode string

const (
	ModeAuto   Mode = "auto"
	ModeAlways Mode = "always"
	ModeNever  Mode = "never"
)

const (
	EnvMode      = "MACONDO_WMP_MODE"
	EnvMaxMB     = "MACONDO_WMP_MAX_MB"
	EnvMaxReadMB = "MACONDO_WMP_MAX_READ_MB"

	defaultMaxMB     = 400
	defaultMaxReadMB = 0
)

// ErrDisabled is returned when policy says not to use a word map for a
// lexicon. Callers treat it the same as a missing word map: fall back to the
// KWG move generator.
var ErrDisabled = errors.New("word map disabled by policy")

// Policy is the resolved word map policy for this process.
type Policy struct {
	Mode Mode
	// MaxBytes is the largest word map that will be used at all, and
	// MaxReadBytes the largest that will be read from disk instead of
	// rebuilt. Zero means no limit. Both apply in ModeAuto only.
	MaxBytes     int64
	MaxReadBytes int64
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
		log.Info().Str("mode", string(p.Mode)).
			Int64("max_mb", p.MaxBytes/(1<<20)).
			Int64("max_read_mb", p.MaxReadBytes/(1<<20)).
			Msg("wmp-policy")
	})
	policyMu.RLock()
	defer policyMu.RUnlock()
	return policy
}

// SetPolicy overrides the policy for this process, for a caller that
// configures itself by some means other than the environment.
func SetPolicy(p Policy) {
	CurrentPolicy() // make sure the once-only env read doesn't clobber this later
	policyMu.Lock()
	policy = p
	policyMu.Unlock()
}

func policyFromEnv() Policy {
	p := Policy{
		Mode:         ModeAuto,
		MaxBytes:     defaultMaxMB << 20,
		MaxReadBytes: defaultMaxReadMB << 20,
	}
	switch strings.ToLower(strings.TrimSpace(os.Getenv(EnvMode))) {
	case "":
	case string(ModeAuto):
	case string(ModeAlways):
		p.Mode = ModeAlways
	case string(ModeNever):
		p.Mode = ModeNever
	default:
		log.Warn().Str(EnvMode, os.Getenv(EnvMode)).
			Msg("unrecognized word map mode; using auto")
	}
	p.MaxBytes = mbFromEnv(EnvMaxMB, defaultMaxMB)
	p.MaxReadBytes = mbFromEnv(EnvMaxReadMB, defaultMaxReadMB)
	return p
}

func mbFromEnv(key string, def int64) int64 {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def << 20
	}
	mb, err := strconv.ParseInt(v, 10, 64)
	if err != nil || mb < 0 {
		log.Warn().Str(key, v).Int64("using", def).
			Msg("word map size limit is not a non-negative number of megabytes")
		return def << 20
	}
	return mb << 20
}

// useWordMap reports whether a word map should be used at all.
func (p Policy) useWordMap() bool { return p.Mode != ModeNever }

// allowsSize reports whether a word map of the given size may be used.
func (p Policy) allowsSize(size int64) bool {
	if p.Mode != ModeAuto || p.MaxBytes == 0 {
		return true
	}
	return size <= p.MaxBytes
}

// prefersRebuild reports whether an on-disk word map of the given size should
// be rebuilt in memory rather than read. Reading is faster at every size we
// have measured, so this is off by default; it exists for a deployment that
// would rather spend CPU than the EFS burst credits a read draws down.
func (p Policy) prefersRebuild(size int64) bool {
	if p.Mode != ModeAuto || p.MaxReadBytes == 0 {
		return false
	}
	return size > p.MaxReadBytes
}
