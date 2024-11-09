package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/rs/zerolog/log"
	"github.com/spf13/viper"

	wglconfig "github.com/domino14/word-golib/config"
)

// Known config names
const (
	ConfigDebug                     string = "debug"
	ConfigKWGPathPrefix                    = "kwg-path-prefix"
	ConfigDataPath                         = "data-path"
	ConfigDefaultLexicon                   = "default-lexicon"
	ConfigDefaultLetterDistribution        = "default-letter-distribution"
	ConfigDefaultBoardLayout               = "default-board-layout"
	ConfigTtableMemFraction                = "ttable-mem-fraction"
	ConfigLambdaFunctionName               = "lambda-function-name"
	ConfigNatsURL                          = "nats-url"
	ConfigWolgesAwsmUrl                    = "wolges-awsm-url"
	ConfigCPUProfile                       = "cpu-profile"
	ConfigMEMProfile                       = "mem-profile"
)

type Config struct {
	sync.Mutex

	viper.Viper

	configPath string
	wglconfig  *wglconfig.Config
}

func DefaultConfig() *Config {
	c := &Config{}
	c.Viper = *viper.New()

	replacer := strings.NewReplacer("-", "_")
	c.SetEnvKeyReplacer(replacer)

	c.AutomaticEnv()
	c.SetEnvPrefix("macondo")

	c.SetDefault(ConfigDefaultLexicon, "NWL23")
	c.SetDefault(ConfigDefaultLetterDistribution, "English")
	c.SetDefault(ConfigTtableMemFraction, 0.25)
	c.SetDefault(ConfigDefaultBoardLayout, "CrosswordGame")
	// Read from an env var MACONDO_DATA_PATH. This might not be a good way to do it:
	c.SetDefault(ConfigDataPath, c.GetString(ConfigDataPath))

	return c
}

func (c *Config) Load(args []string) error {
	c.Viper = *viper.New()
	c.SetConfigName("config")
	c.SetConfigType("yaml")
	// If no config file is found:
	c.SetEnvPrefix("macondo")
	// allow env vars to be specified with `_` instead of `-`
	replacer := strings.NewReplacer("-", "_")
	c.SetEnvKeyReplacer(replacer)

	// Explicitly bind env vars.
	c.BindEnv(ConfigDataPath)
	c.BindEnv(ConfigDefaultLexicon)
	c.BindEnv(ConfigDefaultLetterDistribution)
	c.BindEnv(ConfigTtableMemFraction)
	c.BindEnv(ConfigLambdaFunctionName)
	c.BindEnv(ConfigNatsURL)
	c.BindEnv(ConfigWolgesAwsmUrl)
	c.BindEnv(ConfigDebug)
	c.BindEnv(ConfigKWGPathPrefix)
	c.BindEnv(ConfigCPUProfile)
	c.BindEnv(ConfigMEMProfile)
	c.BindEnv(ConfigDefaultBoardLayout)

	cfgdir, err := os.UserConfigDir()
	if err != nil {
		log.Err(err).Msg("could not get cfgdir; falling back to environment variables")
	} else {
		cfgdir = filepath.Join(cfgdir, "macondo")

		c.AddConfigPath(cfgdir)
		c.configPath = cfgdir
		err = c.ReadInConfig()
		if err != nil {
			if verr, ok := err.(viper.ConfigFileNotFoundError); ok {
				// it's ok if config file is not found. fall back to environment.
				log.Err(verr).Msg("no config file found; falling back to environment variables")
			} else {
				panic(fmt.Errorf("fatal error config file: %w", err))
			}
		}
		// Handle upgrades of macondo by checking to see if data path still exists,
		// if we are reading from a file.
		recreateDataPath := false
		dp := c.GetString(ConfigDataPath)
		_, err := os.Stat(dp)
		if err != nil {
			log.Err(err).Msg("error-checking-for-datapath")
			recreateDataPath = true
		}
		if os.IsNotExist(err) {
			recreateDataPath = true
		}
		if recreateDataPath {
			log.Info().Str("old-data-path", dp).Msg("Did not find old data path. Recreating.")
			c.Set(ConfigDataPath, "./data")
		}
	}

	c.SetDefault(ConfigDataPath, "./data") // will be fixed by toAbsPath below if unspecified.
	c.SetDefault(ConfigDefaultLexicon, "NWL23")
	c.SetDefault(ConfigDefaultLetterDistribution, "English")
	c.SetDefault(ConfigTtableMemFraction, 0.25)
	c.SetDefault(ConfigDefaultBoardLayout, "CrosswordGame")

	return nil
}

// the WGLConfig is a config used for the word-golib library. It is a
// reduced version of this overall config. We calculate it here to avoid
// passing this entire config to that library (and causing a circular import as well).
// This sub-config is not meant to change after we first start this program.
func (c *Config) WGLConfig() *wglconfig.Config {
	if c.wglconfig == nil {
		c.Lock()
		defer c.Unlock()
		c.wglconfig = &wglconfig.Config{
			DataPath:      c.GetString(ConfigDataPath),
			KWGPathPrefix: c.GetString(ConfigKWGPathPrefix),
		}
	}
	return c.wglconfig
}

func (c *Config) AdjustRelativePaths(basepath string) {
	basepath = FindBasePath(basepath)
	absPath := toAbsPath(basepath, c.GetString(ConfigDataPath), "datapath")
	log.Info().Str("absPath", absPath).Msg("setting absolute data path")
	c.Set(ConfigDataPath, absPath)
}

func (c *Config) Write() error {
	werr := c.WriteConfig()
	if _, ok := werr.(viper.ConfigFileNotFoundError); ok {

		if _, err := os.Stat(c.configPath); os.IsNotExist(err) {
			// Directory does not exist, create it
			err = os.Mkdir(c.configPath, 0700)
			if err != nil {
				return err
			}
			log.Info().Msgf("Directory created: %s", c.configPath)
		} else if err != nil {
			return err
		}
		return c.WriteConfigAs(filepath.Join(c.configPath, "config.yaml"))
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
