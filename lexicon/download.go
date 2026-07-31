package lexicon

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/rs/zerolog/log"

	wglconfig "github.com/domino14/word-golib/config"
)

// wordGraphURLBase is the directory in the liwords repo that publishes the
// word graphs macondo consumes. It holds both .kwg (gaddag) and .kad (alpha
// dawg, for WordSmog) files for every supported lexicon.
const wordGraphURLBase = "https://github.com/woogles-io/liwords/raw/refs/heads/master/liwords-ui/public/wasm/2024/"

// EnsureWordGraphFile checks whether the word graph for lexname with the given
// extension (".kwg", ".kad") is present in the data directory and, if not,
// downloads it. ext must include the leading dot.
func EnsureWordGraphFile(lexname, ext string, cfg *wglconfig.Config) error {
	fullpath := filepath.Join(cfg.DataPath, "lexica", "gaddag", cfg.KWGPathPrefix,
		lexname+ext)
	if _, err := os.Stat(fullpath); err == nil {
		return nil // already present
	}
	log.Info().Str("lexicon", lexname).Str("ext", ext).
		Msg("word graph not found; attempting to download...")
	url := wordGraphURLBase + lexname + ext

	out, err := os.CreateTemp(cfg.DataPath, "temp-*"+ext)
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
	// CreateTemp makes the file 0600; word graphs are shared read-only data
	// and may be read by a different user than the one that downloaded them.
	if err = os.Chmod(out.Name(), 0644); err != nil {
		log.Warn().Err(err).Msg("could not set permissions on downloaded word graph")
	}
	if err = os.Rename(out.Name(), fullpath); err != nil {
		return fmt.Errorf("failed to rename file: %w", err)
	}
	log.Info().Str("path", fullpath).Msg("lexicon word graph successfully downloaded")
	return nil
}
