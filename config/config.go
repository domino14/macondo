package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/rs/zerolog/log"
	"github.com/spf13/viper"
)

// Known config names
const (
	ConfigDebug                            = "debug"
	ConfigDataPath                         = "data-path"
	ConfigDefaultLexicon            string = "default-lexicon"
	ConfigDefaultLetterDistribution        = "default-letter-distribution"
	ConfigTtableMemFraction                = "ttable-mem-fraction"
	ConfigLambdaFunctionName               = "lambda-function-name"
	ConfigNatsURL                          = "nats-url"
	ConfigWolgesAwsmUrl                    = "wolges-awsm-url"
)

type Config struct {
	viper.Viper
}

func DefaultConfig() Config {
	var c Config
	c.Viper = *viper.New()

	replacer := strings.NewReplacer("-", "_")
	c.SetEnvKeyReplacer(replacer)

	c.AutomaticEnv()
	c.SetEnvPrefix("macondo")

	c.SetDefault(ConfigDefaultLexicon, "NWL20")
	c.SetDefault(ConfigDefaultLetterDistribution, "English")
	c.SetDefault(ConfigTtableMemFraction, 0.25)
	// Read from an env var MACONDO_DATA_PATH. This might not be a good way to do it:
	c.SetDefault(ConfigDataPath, c.GetString(ConfigDataPath))

	return c
}

func (c *Config) Load(args []string) error {
	c.Viper = *viper.New()
	c.SetConfigName("macondo_config")
	c.SetConfigType("yaml")

	c.AddConfigPath(".")
	c.AddConfigPath("$HOME/.macondo")
	c.AddConfigPath("/etc/macondo/")
	err := c.ReadInConfig()
	if err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			// it's ok if config file is not found. fall back to environment.
			log.Info().Msg("no config file found; falling back to environment variables")
		} else {
			panic(fmt.Errorf("fatal error config file: %w", err))
		}
	}
	c.SetEnvPrefix("macondo")
	replacer := strings.NewReplacer("-", "_")
	c.SetEnvKeyReplacer(replacer)

	c.BindEnv(ConfigDataPath)
	c.BindEnv(ConfigDefaultLexicon)
	c.BindEnv(ConfigDefaultLetterDistribution)
	c.BindEnv(ConfigTtableMemFraction)
	c.BindEnv(ConfigLambdaFunctionName)
	c.BindEnv(ConfigNatsURL)
	c.BindEnv(ConfigWolgesAwsmUrl)
	c.BindEnv(ConfigDebug)
	// allow env vars to be specified with `_` instead of `-`

	c.SetDefault(ConfigDefaultLexicon, "NWL20")
	c.SetDefault(ConfigDefaultLetterDistribution, "English")
	c.SetDefault(ConfigTtableMemFraction, 0.25)

	return nil
}

func (c *Config) AdjustRelativePaths(basepath string) {
	basepath = FindBasePath(basepath)
	c.Set(ConfigDataPath, toAbsPath(basepath, c.GetString(ConfigDataPath), "datapath"))
}

func (c *Config) Write() error {
	werr := c.WriteConfig()
	if werr != nil {
		if _, ok := werr.(viper.ConfigFileNotFoundError); ok {
			return c.WriteConfigAs("./macondo_config.yaml")
		}
	}
	return werr
}

func FindBasePath(path string) string {
	// Search up a path until we find the toplevel dir with data/ under it.
	// This will likely do bad things if there is no such dir but right now we
	// are running stuff from within the macondo directory and ultimately we want
	// to use something like $HOME/.macondo anyway rather than the exe path.
	oldpath := ""
	for oldpath != path {
		data := filepath.Join(path, "data")
		_, err := os.Stat(data)
		if !(os.IsNotExist(err)) {
			log.Debug().Str("data", data).Msg("found data dir")
			break
		}
		oldpath = path
		path = filepath.Dir(path)
	}
	// Prevent the function from infinite-looping. If we make it all the way to
	// the top then exit anyway.
	log.Debug().Str("basepath", path).Msg("returning base-path")
	return path
}

func toAbsPath(basepath string, path string, logname string) string {
	if strings.HasPrefix(path, "./") {
		path = filepath.Join(basepath, path)
		log.Info().Str(logname, path).Msgf("adjusted relative path")
	}
	return path
}
