package turnplayer

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/rs/zerolog/log"

	wglconfig "github.com/domino14/word-golib/config"
	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

type Lexicon struct {
	Name         string
	Distribution string
}

func (lex *Lexicon) ToDisplayString() string {
	return fmt.Sprintf("%s [%s]", lex.Name, lex.Distribution)
}

type GameOptions struct {
	Lexicon         *Lexicon
	ChallengeRule   pb.ChallengeRule
	BoardLayoutName string
	Variant         game.Variant
}

func (opts *GameOptions) SetDefaults(cfg *config.Config) {
	if opts.Lexicon == nil {
		opts.Lexicon = &Lexicon{cfg.GetString(config.ConfigDefaultLexicon), cfg.GetString(config.ConfigDefaultLetterDistribution)}
		log.Info().Msgf("using default lexicon %v", opts.Lexicon)
	}
	if opts.BoardLayoutName == "" {
		opts.BoardLayoutName = cfg.GetString(config.ConfigDefaultBoardLayout)
		log.Info().Msgf("using default board layout %v", opts.BoardLayoutName)
	}
	if opts.Variant == "" {
		opts.Variant = game.VarClassic
	}
}

// EnsureKWG checks whether the KWG file for lexname is present and, if not,
// downloads it from the woogles-io/liwords repository. It is safe to call
// before any code that calls kwg.GetKWG so that load paths other than
// "set lexicon" also get the file-not-found download behaviour.
func EnsureKWG(lexname string, cfg *wglconfig.Config) error {
	fullpath := filepath.Join(cfg.DataPath, "lexica", "gaddag", cfg.KWGPathPrefix,
		lexname+".kwg")
	if _, err := os.Stat(fullpath); err == nil {
		return nil // already present
	}
	log.Info().Str("lexicon", lexname).Msg("KWG not found; attempting to download...")
	url := "https://github.com/woogles-io/liwords/raw/refs/heads/master/liwords-ui/public/wasm/2024/" + lexname + ".kwg"

	out, err := os.CreateTemp(cfg.DataPath, "temp-*.kwg")
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer func() {
		os.Remove(out.Name())
	}()
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download file: received status code %d", resp.StatusCode)
	}
	if _, err = io.Copy(out, resp.Body); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}
	out.Close()
	if err = os.Rename(out.Name(), fullpath); err != nil {
		return fmt.Errorf("failed to rename file: %w", err)
	}
	log.Info().Str("path", fullpath).Msg("lexicon word graph successfully downloaded")
	return nil
}

func (opts *GameOptions) SetLexicon(fields []string, cfg *wglconfig.Config) error {
	lexname := ""
	letdist := ""
	var err error
	if len(fields) == 1 {
		lexname = fields[0]
		letdist, err = tilemapping.ProbableLetterDistributionName(lexname)
		if err != nil {
			log.Err(err).Msg("letter distribution not found for this lexicon; assuming english")
			letdist = "english"
		} else {
			log.Info().Msgf("assuming letter distribution %v", letdist)
		}
	} else if len(fields) == 2 {
		lexname, letdist = fields[0], fields[1]
	} else {
		return errors.New("Valid formats are 'lexicon' and 'lexicon alphabet'")
	}
	lexname = strings.ToUpper(lexname)
	if err := EnsureKWG(lexname, cfg); err != nil {
		return err
	}
	opts.Lexicon = &Lexicon{Name: lexname, Distribution: letdist}
	return nil
}

func (opts *GameOptions) SetChallenge(rule string) error {
	val, err := ParseChallengeRule(rule)
	if err != nil {
		return err
	}
	opts.ChallengeRule = val
	return nil
}

func (opts *GameOptions) SetBoardLayoutName(name string) error {
	switch name {
	case board.CrosswordGameLayout, board.SuperCrosswordGameLayout:
		opts.BoardLayoutName = name
	default:
		return fmt.Errorf("%v is not a supported board layout", name)
	}
	return nil
}

func (opts *GameOptions) SetVariant(name string) error {
	switch name {
	case string(game.VarClassic):
		opts.Variant = game.VarClassic
	case string(game.VarClassicSuper):
		opts.Variant = game.VarClassicSuper
	case string(game.VarWordSmog):
		opts.Variant = game.VarWordSmog
	case string(game.VarWordSmogSuper):
		opts.Variant = game.VarWordSmogSuper
	default:
		return fmt.Errorf("%v is not a supported variant name", name)
	}
	return nil
}
