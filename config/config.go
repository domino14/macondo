package config

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/namsral/flag"
	"github.com/rs/zerolog/log"
)

type Config struct {
	Debug                     bool
	LetterDistributionPath    string
	StrategyParamsPath        string
	LexiconPath               string
	DefaultLexicon            string
	DefaultLetterDistribution string
}

func (c *Config) Load(args []string) error {
	fs := flag.NewFlagSet("macondo", flag.ContinueOnError)
	fs.BoolVar(&c.Debug, "debug", false, "debug logging on")
	fs.StringVar(&c.StrategyParamsPath, "strategy-params-path", "./data/strategy", "directory holding strategy files")
	fs.StringVar(&c.LetterDistributionPath, "letter-distribution-path", "./data/letterdistributions", "directory holding letter distribution files")
	fs.StringVar(&c.LexiconPath, "lexicon-path", "./data/lexica", "directory holding lexicon files")
	fs.StringVar(&c.DefaultLexicon, "default-lexicon", "NWL18", "the default lexicon to use")
	fs.StringVar(&c.DefaultLetterDistribution, "default-letter-distribution", "English", "the default letter distribution to use. English, EnglishSuper, Spanish, Polish, etc.")
	err := fs.Parse(args)
	return err
}

func (c *Config) AdjustRelativePaths(basepath string) {
	basepath = FindBasePath(basepath)
	c.LexiconPath = toAbsPath(basepath, c.LexiconPath, "lexiconpath")
	c.StrategyParamsPath = toAbsPath(basepath, c.StrategyParamsPath, "sppath")
	c.LetterDistributionPath = toAbsPath(basepath, c.LetterDistributionPath, "ldpath")
}

func FindBasePath(path string) string {
	// Search up a path until we find the toplevel dir with data/ under it.
	// This will likely do bad things if there is no such dir but right now we
	// are running stuff from within the macondo directory and ultimately we want
	// to use something like $HOME/.macondo anyway rather than the exe path.
	for {
		data := filepath.Join(path, "data")
		_, err := os.Stat(data)
		if !(os.IsNotExist(err)) {
			break
		}
		path = filepath.Dir(path)
	}
	return path
}

func toAbsPath(basepath string, path string, logname string) string {
	if strings.HasPrefix(path, "./") {
		path = filepath.Join(basepath, path)
		log.Info().Str(logname, path).Msgf("adjusted relative path")
	}
	return path
}
