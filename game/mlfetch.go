package game

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/dataloaders"
)

// The shipped value nets are not in this repository: they are published on
// Hugging Face (ml-models-url) and downloaded the first time a game needs
// one, into strategy/default/models/<name>/<version>/model.onnx, the layout
// Triton serves. Next to each downloaded file, model.onnx.source records the
// URL it came from; when ml-models-url moves to another revision the file is
// downloaded again. A model.onnx with no .source file (one trained here, or
// copied in by hand) is always used as it is.

// ErrNoMLModel means no such model is on disk or published.
var ErrNoMLModel = errors.New("no such model")

var mlFetchMu sync.Mutex

func mlModelFile(name, version string) string {
	return fmt.Sprintf("models/%s/%s/model.onnx", name, version)
}

// ensureMLModel makes the model's ONNX file available in the strategy
// directory, downloading it if it is missing or out of date.
func ensureMLModel(cfg *config.Config, name, version, lexicon string) error {
	strat := dataloaders.StrategyParamsPath(cfg.WGLConfig())
	rel := mlModelFile(name, version)
	dst := filepath.Join(strat, "default", filepath.FromSlash(rel))
	url := ""
	if base := strings.TrimSuffix(cfg.GetString(config.ConfigMLModelsURL), "/"); base != "" {
		url = base + "/" + rel[len("models/"):]
	}

	mlFetchMu.Lock()
	defer mlFetchMu.Unlock()
	stale := false
	if src, err := os.ReadFile(dst + ".source"); err == nil {
		if url == "" || string(src) == url {
			return nil
		}
		stale = true
	} else if r, err := dataloaders.StratFileForLexicon(strat, rel, lexicon); err == nil {
		r.Close()
		return nil
	}
	if url == "" {
		return fmt.Errorf("%s: %w (ml-models-url is empty, so nothing is downloaded)", name, ErrNoMLModel)
	}
	err := downloadMLModel(url, dst)
	if err != nil && stale && !errors.Is(err, ErrNoMLModel) {
		log.Warn().Err(err).Str("model", name).Msg("could not update the model; using the copy on disk")
		return nil
	}
	return err
}

func downloadMLModel(url, dst string) error {
	log.Info().Str("url", url).Msg("downloading model")
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("downloading %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("%s: %w", url, ErrNoMLModel)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("downloading %s: status %d", url, resp.StatusCode)
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	out, err := os.CreateTemp(filepath.Dir(dst), "model-*.onnx")
	if err != nil {
		return err
	}
	defer os.Remove(out.Name())
	written, err := io.Copy(out, resp.Body)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("downloading %s: %w", url, err)
	}
	// CreateTemp makes the file 0600; the data directory may be shared.
	if err := os.Chmod(out.Name(), 0o644); err != nil {
		log.Warn().Err(err).Msg("could not set permissions on the downloaded model")
	}
	if err := os.Rename(out.Name(), dst); err != nil {
		return err
	}
	if err := os.WriteFile(dst+".source", []byte(url), 0o644); err != nil {
		return err
	}
	log.Info().Str("path", dst).Int64("bytes", written).Msg("model downloaded")
	return nil
}
