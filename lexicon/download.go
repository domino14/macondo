package lexicon

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/rs/zerolog/log"

	wglconfig "github.com/domino14/word-golib/config"
)

// lexiconFileURLBase is the directory in the liwords repo that publishes the
// per-lexicon data macondo consumes: .kwg (gaddag), .kad (alpha dawg, for
// WordSmog) and .klv2 (leave values) files.
const lexiconFileURLBase = "https://github.com/woogles-io/liwords/raw/refs/heads/master/liwords-ui/public/wasm/2024/"

// ErrNotPublished means no such file is published. Plenty of per-lexicon data
// is optional -- an old lexicon may have no leave values, and only a handful
// have a SuperCrosswordGame set -- so callers should treat this as an answer
// rather than as a failure.
var ErrNotPublished = errors.New("no such lexicon data file is published")

// EnsureLexiconFile checks whether the per-lexicon data file for lexname with
// the given extension (".kwg", ".kad", ".klv2") is present in the data
// directory and, if not, downloads it. ext must include the leading dot.
func EnsureLexiconFile(lexname, ext string, cfg *wglconfig.Config) error {
	fullpath := filepath.Join(cfg.DataPath, "lexica", "gaddag", cfg.KWGPathPrefix,
		lexname+ext)
	if fi, err := os.Stat(fullpath); err == nil {
		log.Info().Str("file", lexname+ext).Str("path", fullpath).Int64("bytes", fi.Size()).
			Msg("lexicon data file found on disk")
		return nil
	}
	url := lexiconFileURLBase + lexname + ext
	log.Info().Str("file", lexname+ext).Str("url", url).
		Msg("lexicon data file not on disk; downloading...")

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
	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("%s%s: %w", lexname, ext, ErrNotPublished)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download file: received status code %d", resp.StatusCode)
	}
	written, err := io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}
	out.Close()
	// CreateTemp makes the file 0600; these are shared read-only data files
	// and may be read by a different user than the one that downloaded them.
	if err = os.Chmod(out.Name(), 0644); err != nil {
		log.Warn().Err(err).Msg("could not set permissions on downloaded lexicon file")
	}
	if err = os.Rename(out.Name(), fullpath); err != nil {
		return fmt.Errorf("failed to rename file: %w", err)
	}
	log.Info().Str("file", lexname+ext).Str("path", fullpath).Int64("bytes", written).
		Msg("lexicon data file downloaded")
	return nil
}
