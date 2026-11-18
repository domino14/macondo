package lexicon

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog/log"

	wglconfig "github.com/domino14/word-golib/config"
)

// Per-lexicon data -- .kwg (gaddag), .kad (alpha dawg, for WordSmog) and
// .klv2 (leave values) files -- is not in this repository. Woogles publishes
// it in a bucket, and each file there carries its sha256 as metadata. A file
// is downloaded when it is missing, and downloaded again when the published
// one no longer has the sha256 of the copy on disk, so a rebuilt word graph
// or a new leave table reaches an install the next time it starts.

const (
	// DefaultLexiconFileURL is the directory the files are published in.
	DefaultLexiconFileURL = "https://woogles.io/wasm/2026/"
	// LexiconFileURLEnv names the environment variable that replaces
	// DefaultLexiconFileURL.
	LexiconFileURLEnv = "MACONDO_LEXICON_URL"
	// LexiconAutoUpdateEnv names the environment variable that, set to
	// "false", "0" or "off", leaves a file that is on disk alone: only
	// missing files are downloaded.
	LexiconAutoUpdateEnv = "MACONDO_LEXICON_AUTO_UPDATE"

	// shaHeader is where the bucket keeps a file's sha256.
	shaHeader = "x-amz-meta-sha256"
)

// ErrNotPublished means no such file is published. Plenty of per-lexicon data
// is optional -- an old lexicon may have no leave values, and only a handful
// have a SuperCrosswordGame set -- so callers should treat this as an answer
// rather than as a failure.
var ErrNotPublished = errors.New("no such lexicon data file is published")

var (
	headClient     = &http.Client{Timeout: 20 * time.Second}
	downloadClient = &http.Client{Timeout: 10 * time.Minute}
)

// ensureTried remembers the answer EnsureLexiconFile gave for a file, keyed by
// name+extension, so EnsureLexiconFileOnce can repeat it without touching the
// disk or the network. A nil value means the file was there (or was fetched).
var ensureTried sync.Map

// ensured holds the paths this process has already checked against the
// published files (or fetched), each with the mutex that keeps two goroutines
// from fetching the same file at once.
var ensured sync.Map

type ensureState struct {
	sync.Mutex
	done bool
}

func lexiconFileURL(name string) string {
	base := os.Getenv(LexiconFileURLEnv)
	if base == "" {
		base = DefaultLexiconFileURL
	}
	if !strings.HasSuffix(base, "/") {
		base += "/"
	}
	return base + name
}

func autoUpdate() bool {
	switch strings.ToLower(os.Getenv(LexiconAutoUpdateEnv)) {
	case "false", "0", "off", "no":
		return false
	}
	return true
}

// EnsureLexiconFileOnce is EnsureLexiconFile, except that it is attempted at
// most once per file per process and every call after the first repeats the
// first one's answer. Callers on a hot path want this: a bot that needs a file
// which isn't published, or which can't be written to disk, would otherwise
// make an HTTP request every single turn.
func EnsureLexiconFileOnce(lexname, ext string, cfg *wglconfig.Config) error {
	key := lexname + ext
	if v, tried := ensureTried.Load(key); tried {
		err, _ := v.(error) // v is nil, and so err is nil, on the success path
		return err
	}
	err := EnsureLexiconFile(lexname, ext, cfg)
	ensureTried.Store(key, err)
	return err
}

// EnsureLexiconFile makes sure the per-lexicon data file for lexname with the
// given extension (".kwg", ".kad", ".klv2") is in the data directory and is
// the published one. ext must include the leading dot.
//
// A missing file is downloaded. A file on disk is compared, by sha256, with
// the published one and replaced if they differ. A file on disk is never a
// reason to fail: when nothing by that name is published (a private lexicon,
// an experimental leave table), or the published file cannot be checked or
// fetched (no network, a read-only data directory), the copy on disk is used.
//
// Once a file has been found current or fetched, later calls in the same
// process return at once; a call that failed is tried again.
func EnsureLexiconFile(lexname, ext string, cfg *wglconfig.Config) error {
	name := lexname + ext
	dir := filepath.Join(cfg.DataPath, "lexica", "gaddag", cfg.KWGPathPrefix)
	fullpath := filepath.Join(dir, name)

	v, _ := ensured.LoadOrStore(fullpath, &ensureState{})
	state := v.(*ensureState)
	state.Lock()
	defer state.Unlock()
	if state.done {
		return nil
	}

	_, statErr := os.Stat(fullpath)
	onDisk := statErr == nil
	if onDisk && !autoUpdate() {
		state.done = true
		return nil
	}

	url := lexiconFileURL(name)
	want, err := publishedSHA(url)
	if err != nil {
		if !onDisk {
			if errors.Is(err, ErrNotPublished) {
				return fmt.Errorf("%s: %w", name, ErrNotPublished)
			}
			return fmt.Errorf("%s is not on disk and could not be downloaded: %w", name, err)
		}
		if errors.Is(err, ErrNotPublished) {
			// Somebody's own file; nothing to compare it with.
			state.done = true
		} else {
			log.Warn().Err(err).Str("file", name).
				Msg("could not check for a newer lexicon data file; using the copy on disk")
		}
		return nil
	}
	if onDisk {
		have, err := fileSHA(fullpath)
		if err == nil && have == want {
			log.Debug().Str("file", name).Msg("lexicon data file is current")
			state.done = true
			return nil
		}
		log.Info().Str("file", name).Str("url", url).
			Msg("a different lexicon data file is published; downloading it...")
	} else {
		log.Info().Str("file", name).Str("url", url).
			Msg("lexicon data file not on disk; downloading...")
	}

	written, err := download(url, want, dir, fullpath, ext)
	if err != nil {
		if onDisk {
			log.Warn().Err(err).Str("file", name).
				Msg("could not update the lexicon data file; using the copy on disk")
			return nil
		}
		return err
	}
	state.done = true
	log.Info().Str("file", name).Str("path", fullpath).Int64("bytes", written).
		Msg("lexicon data file downloaded")
	if onDisk && ext == ".kwg" {
		// A word map (.wmp) is built from the word graph and kept beside the
		// lexica folder. One built from the graph just replaced would go on
		// answering for the old word list, so it goes; it is built again from
		// the new graph when it is next wanted.
		wmp := filepath.Join(cfg.DataPath, "lexica", lexname+".wmp")
		if err := os.Remove(wmp); err == nil {
			log.Info().Str("path", wmp).Msg("removed the word map built from the old word graph")
		} else if !os.IsNotExist(err) {
			log.Warn().Err(err).Str("path", wmp).
				Msg("could not remove the word map built from the old word graph")
		}
	}
	return nil
}

// published reports whether a response is a published file. The site answers
// a path it has no file for with its own page (status 200, text/html), so a
// file only counts as published when it comes with the sha256 the bucket
// keeps for it.
func published(resp *http.Response) (sha string, err error) {
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusForbidden {
		return "", ErrNotPublished
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("received status code %d", resp.StatusCode)
	}
	sha = strings.ToLower(resp.Header.Get(shaHeader))
	if sha == "" || strings.HasPrefix(resp.Header.Get("Content-Type"), "text/html") {
		return "", ErrNotPublished
	}
	return sha, nil
}

func publishedSHA(url string) (string, error) {
	resp, err := headClient.Head(url)
	if err != nil {
		return "", err
	}
	resp.Body.Close()
	return published(resp)
}

func fileSHA(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// download fetches url into fullpath, by way of a temporary file beside it
// that only takes its place once it is whole and has the sha256 want.
func download(url, want, dir, fullpath, ext string) (int64, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return 0, fmt.Errorf("failed to create directory: %w", err)
	}
	out, err := os.CreateTemp(dir, ".download-*"+ext)
	if err != nil {
		return 0, fmt.Errorf("failed to create file: %w", err)
	}
	defer func() {
		out.Close()
		os.Remove(out.Name())
	}()
	resp, err := downloadClient.Get(url)
	if err != nil {
		return 0, fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()
	if _, err := published(resp); err != nil {
		return 0, fmt.Errorf("failed to download %s: %w", url, err)
	}
	h := sha256.New()
	written, err := io.Copy(io.MultiWriter(out, h), resp.Body)
	if err != nil {
		return 0, fmt.Errorf("failed to write file: %w", err)
	}
	if err := out.Close(); err != nil {
		return 0, fmt.Errorf("failed to write file: %w", err)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return 0, fmt.Errorf("downloaded %s has sha256 %s, not the published %s", url, got, want)
	}
	// CreateTemp makes the file 0600; these are shared read-only data files
	// and may be read by a different user than the one that downloaded them.
	if err := os.Chmod(out.Name(), 0644); err != nil {
		log.Warn().Err(err).Msg("could not set permissions on downloaded lexicon file")
	}
	if err := os.Rename(out.Name(), fullpath); err != nil {
		return 0, fmt.Errorf("failed to rename file: %w", err)
	}
	return written, nil
}
