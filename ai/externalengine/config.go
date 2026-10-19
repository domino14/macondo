package externalengine

// Config holds the configuration for an external engine player.
// Any engine implementing the standard CLI/JSON contract
// (--gcg, --lexicon, --rack → one JSON line) can be configured here.
type Config struct {
	// BinaryPath is the full path to the external engine binary.
	BinaryPath string
	// ExtraArgs are optional arguments prepended before the standard flags.
	ExtraArgs []string
}
