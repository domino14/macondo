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

// puzzleSetKeepFile points keeping at a file and loads what is already in it, so
// that reopening an old collection does not duplicate its puzzles.
func (sc *ShellController) puzzleSetKeepFile(path string) error {
	full := expandHomePath(path)

	// Appending the file you are reading would grow it under you and duplicate
	// whatever you kept on a later pass.
	if openPath := expandHomePath(sc.puzzleFile); openPath != "" {
		a, errA := filepath.Abs(full)
		b, errB := filepath.Abs(openPath)
		if errA == nil && errB == nil && a == b {
			return fmt.Errorf("that is the file you are browsing; keep into a different one")
		}
	}

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
		dest = pgDefaultKeepFile(sc.puzzleFile)
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

// puzzleUnkeep takes the current puzzle back out of the keep file. Keeping is
// one keystroke and easy to do before running `gen` and changing your mind, so
// it needs an undo; the file is rewritten without that record.
func (sc *ShellController) puzzleUnkeep() (*Response, error) {
	if sc.puzzleKeepFile == "" {
		return nil, fmt.Errorf("no keep file is open")
	}
	rec := sc.puzzleSet[sc.puzzleIdx]
	key := pgKeepKey(rec)
	if !sc.puzzleKept[key] {
		return nil, fmt.Errorf("puzzle %d is not in %s", sc.puzzleIdx+1, sc.puzzleKeepFile)
	}

	full := expandHomePath(sc.puzzleKeepFile)
	f, err := os.Open(full)
	if err != nil {
		return nil, err
	}
	var out []byte
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var kept pgRecord
		if err := json.Unmarshal([]byte(line), &kept); err != nil {
			f.Close()
			return nil, fmt.Errorf("%s is not a puzzle file: %w", sc.puzzleKeepFile, err)
		}
		if pgKeepKey(&kept) == key {
			continue
		}
		out = append(out, line...)
		out = append(out, '\n')
	}
	scanErr := scanner.Err()
	f.Close()
	if scanErr != nil {
		return nil, scanErr
	}

	// Write beside the target and rename, so an interrupted rewrite cannot
	// leave a half-written collection behind.
	tmp := full + ".tmp"
	if err := os.WriteFile(tmp, out, 0644); err != nil {
		return nil, err
	}
	if err := os.Rename(tmp, full); err != nil {
		os.Remove(tmp)
		return nil, err
	}

	delete(sc.puzzleKept, key)
	return msg(fmt.Sprintf("Removed puzzle %d from %s (%d kept).",
		sc.puzzleIdx+1, sc.puzzleKeepFile, len(sc.puzzleKept))), nil
}

// puzzleKeptSummary reports where puzzles are being kept and how many are there.
func (sc *ShellController) puzzleKeptSummary() string {
	if sc.puzzleKeepFile == "" {
		return "No keep file yet. Name one with `puzzle keep <file>`."
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
