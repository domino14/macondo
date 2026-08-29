package shell

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Keeping a puzzle appends its record, verbatim, to a second file. That file is
// itself a puzzle file -- `puzzle open` reads it back, and `puzzlegen -out`
// would have written the same bytes -- so a few hundred generated puzzles can be
// browsed down to the handful worth keeping without any conversion step.
//
// The destination is remembered after it is named once, since the whole point is
// to say `puzzle keep` a few dozen times.

// pgKeepKey identifies a puzzle across files. A position is a game and a turn;
// two records with the same pair are the same puzzle however they were
// generated.
func pgKeepKey(rec *pgRecord) string {
	return fmt.Sprintf("%s:%d", rec.GameID, rec.Turn)
}

// pgDefaultKeepFile puts the keepers beside the file they came from and names
// them after it, so that `puzzle keep` works without anyone having to invent a
// filename first. Keeping reports the path it used, so nothing is written
// somewhere the user cannot see.
func pgDefaultKeepFile(openFile string) string {
	if openFile == "" {
		return "kept.jsonl"
	}
	ext := filepath.Ext(openFile)
	return strings.TrimSuffix(openFile, ext) + "-kept" + ext
}

// pgLooksLikeKeepFile reports whether a path is one pgDefaultKeepFile would
// have produced. Browsing a collection and keeping out of it again is a second
// round of curating, and guessing at a name for it gives you
// puzzles-kept-kept.jsonl -- so that case asks instead of guessing.
func pgLooksLikeKeepFile(path string) bool {
	base := filepath.Base(path)
	return strings.HasSuffix(strings.TrimSuffix(base, filepath.Ext(base)), "-kept")
}

// pgSamePath reports whether two paths name the same file.
func pgSamePath(a, b string) bool {
	if a == "" || b == "" {
		return false
	}
	absA, errA := filepath.Abs(expandHomePath(a))
	absB, errB := filepath.Abs(expandHomePath(b))
	return errA == nil && errB == nil && absA == absB
}

// puzzleSetKeepFile points keeping at a file and loads what is already in it, so
// that reopening an old collection does not duplicate its puzzles.
func (sc *ShellController) puzzleSetKeepFile(path string) error {
	full := expandHomePath(path)
	kept := map[string]bool{}
	f, err := os.Open(full)
	if err == nil {
		defer f.Close()
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var rec pgRecord
			if err := json.Unmarshal([]byte(line), &rec); err != nil {
				return fmt.Errorf("%s is not a puzzle file: %w", path, err)
			}
			kept[pgKeepKey(&rec)] = true
		}
		if err := scanner.Err(); err != nil {
			return err
		}
	} else if !os.IsNotExist(err) {
		return err
	}

	sc.puzzleKeepFile = path
	sc.puzzleKept = kept
	return nil
}

// puzzleKeep appends the current puzzle to the keep file, naming that file on
// the first call and reusing it afterwards.
func (sc *ShellController) puzzleKeep(dest string) (*Response, error) {
	if dest == "" && sc.puzzleKeepFile == "" {
		if pgLooksLikeKeepFile(sc.puzzleFile) {
			return nil, fmt.Errorf(
				"%s is already a collection; name where these should go, as `puzzle keep <file>` "+
					"(or `puzzle unkeep` to take one out of it)", sc.puzzleFile)
		}
		dest = pgDefaultKeepFile(sc.puzzleFile)
	}
	// Checked before the destination is recorded, and again on every keep: a
	// remembered destination becomes the open file the moment you open it, and
	// appending there would grow the file under the browser.
	if pgSamePath(dest, sc.puzzleFile) || pgSamePath(sc.puzzleKeepFile, sc.puzzleFile) {
		return nil, fmt.Errorf("that is the file you are browsing; keep into a different one")
	}
	if dest != "" {
		if err := sc.puzzleSetKeepFile(dest); err != nil {
			return nil, err
		}
	}

	rec := sc.puzzleSet[sc.puzzleIdx]
	key := pgKeepKey(rec)
	if sc.puzzleKept[key] {
		return msg(fmt.Sprintf("Puzzle %d is already in %s (%d kept).",
			sc.puzzleIdx+1, sc.puzzleKeepFile, len(sc.puzzleKept))), nil
	}

	line, err := json.Marshal(rec)
	if err != nil {
		return nil, err
	}
	f, err := os.OpenFile(expandHomePath(sc.puzzleKeepFile),
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if _, err := f.Write(append(line, '\n')); err != nil {
		return nil, err
	}

	sc.puzzleKept[key] = true
	return msg(fmt.Sprintf("Kept puzzle %d of %d → %s (%d kept).",
		sc.puzzleIdx+1, len(sc.puzzleSet), sc.puzzleKeepFile, len(sc.puzzleKept))), nil
}

// puzzleUnkeep takes the current puzzle out of the collection it is in.
//
// Which collection that is depends on what you are doing. With a keep file
// named, it is that file -- keeping is one keystroke and easy to do just before
// running `gen` and changing your mind. With no keep file named you are browsing
// a collection rather than building one, and the file to take it out of is the
// one open in front of you; pruning a collection you made earlier is the whole
// reason to open it again.
func (sc *ShellController) puzzleUnkeep() (*Response, error) {
	rec := sc.puzzleSet[sc.puzzleIdx]
	key := pgKeepKey(rec)

	target := sc.puzzleKeepFile
	pruningOpenFile := target == ""
	if pruningOpenFile {
		target = sc.puzzleFile
	} else if !sc.puzzleKept[key] {
		return nil, fmt.Errorf("puzzle %d is not in %s", sc.puzzleIdx+1, sc.puzzleKeepFile)
	}

	removed, err := pgRemoveFromPuzzleFile(expandHomePath(target), key)
	if err != nil {
		return nil, err
	}
	if removed == 0 {
		return nil, fmt.Errorf("puzzle %d is not in %s", sc.puzzleIdx+1, target)
	}
	delete(sc.puzzleKept, key)

	if !pruningOpenFile {
		return msg(fmt.Sprintf("Removed puzzle %d from %s (%d kept).",
			sc.puzzleIdx+1, sc.puzzleKeepFile, len(sc.puzzleKept))), nil
	}

	gone := sc.dropCurrentFromBrowser()
	if len(sc.puzzleSet) == 0 {
		return msg(fmt.Sprintf("Removed puzzle %d from %s. It is now empty.", gone, target)), nil
	}
	sc.showMessage(fmt.Sprintf("Removed puzzle %d from %s, the file you are browsing. %d left.",
		gone, target, len(sc.puzzleSet)))
	return sc.puzzleGoto(sc.puzzleIdx + 1)
}

// dropCurrentFromBrowser forgets the puzzle just deleted from the open file and
// returns its former number. The browser has to lose it too, or its numbering
// stops matching what reopening the file would show.
func (sc *ShellController) dropCurrentFromBrowser() int {
	gone := sc.puzzleIdx + 1
	sc.puzzleSet = append(sc.puzzleSet[:sc.puzzleIdx], sc.puzzleSet[sc.puzzleIdx+1:]...)
	if sc.puzzleIdx >= len(sc.puzzleSet) {
		sc.puzzleIdx = len(sc.puzzleSet) - 1
	}
	if sc.puzzleIdx < 0 {
		sc.puzzleIdx = 0
	}
	return gone
}

// pgRemoveFromPuzzleFile rewrites a puzzle file without the records matching
// key, and reports how many it dropped. The rewrite goes to a temp file and is
// renamed into place, so an interrupted one cannot leave a half-written
// collection behind.
func pgRemoveFromPuzzleFile(path, key string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	var out []byte
	removed := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var rec pgRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			f.Close()
			return 0, fmt.Errorf("%s is not a puzzle file: %w", path, err)
		}
		if pgKeepKey(&rec) == key {
			removed++
			continue
		}
		out = append(out, line...)
		out = append(out, '\n')
	}
	scanErr := scanner.Err()
	f.Close()
	if scanErr != nil {
		return 0, scanErr
	}
	if removed == 0 {
		return 0, nil
	}

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, out, 0644); err != nil {
		return 0, err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return 0, err
	}
	return removed, nil
}

// puzzleKeptSummary reports where puzzles are being kept and how many are there.
func (sc *ShellController) puzzleKeptSummary() string {
	if sc.puzzleKeepFile == "" {
		return fmt.Sprintf(
			"No keep file yet; `puzzle keep <file>` names one. "+
				"`puzzle unkeep` would take a puzzle out of %s itself.", sc.puzzleFile)
	}
	inThisFile := 0
	for _, rec := range sc.puzzleSet {
		if sc.puzzleKept[pgKeepKey(rec)] {
			inThisFile++
		}
	}
	return fmt.Sprintf("Keeping in %s: %d puzzle(s), %d of them from %s.",
		sc.puzzleKeepFile, len(sc.puzzleKept), inThisFile, sc.puzzleFile)
}
