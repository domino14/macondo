package config

import "github.com/namsral/flag"

type Config struct {
	StrategyParamsPath        string
	LexiconPath               string
	DefaultLexicon            string
	DefaultLetterDistribution string
}

func (c *Config) Load(args []string) error {
	fs := flag.NewFlagSet("macondo", flag.ContinueOnError)
	fs.StringVar(&c.StrategyParamsPath, "strategy-params-path", "./data/strategy", "directory holding strategy files")
	fs.StringVar(&c.LexiconPath, "lexicon-path", "./data/lexica", "directory holding lexicon files")
	fs.StringVar(&c.DefaultLexicon, "default-lexicon", "NWL18", "the default lexicon to use")
	fs.StringVar(&c.DefaultLetterDistribution, "default-letter-distribution", "English", "the default letter distribution to use. English, EnglishSuper, Spanish, Polish, etc.")
	err := fs.Parse(args)
	return err
}
