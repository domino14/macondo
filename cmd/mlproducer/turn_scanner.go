package main

import (
	"bufio"
	"encoding/csv"
	"io"
	"strconv"
	"strings"
)

// --------------------------------------------------------------------
// One-line-per-turn text source  →  stream of Turn objects
// --------------------------------------------------------------------
type Turn struct {
	PlayerID       string  // "p1" / "p2"
	GameID         string  // 6840faff…
	TurnNumber     int     // 1, 2, …
	Rack           string  // "DIIORTZ"
	Play           string  // "8E ZITI" or "(exch DGLNPV)"
	Score          int     // points scored this turn
	TotalScore     int     // cumulative score for this player
	TilesPlayed    int     // count in the `play` field (0 for exchange/pass)
	Leave          string  // rack after move  (may be "")
	Equity         float64 // Macondo equity of the move
	TilesRemaining int     // tiles left in bag after this ply
	OppScore       int     // opponent’s cumulative score after the ply
	// You may add helpers like Spread() or PlyIndex() later.
}

// -------------------------------------------------------------------
// TurnScanner: stream-style reader
// -------------------------------------------------------------------
type TurnScanner struct {
	r    *csv.Reader // wraps underlying bufio.Reader
	curr Turn        // last successfully parsed turn
	err  error
}

// NewTurnScanner prepares a CSV reader that:
//   - skips the header row   (playerID,gameID,…)
//   - trims leading spaces in each field
//   - is tolerant of very long lines
func NewTurnScanner(src io.Reader) *TurnScanner {
	br := bufio.NewReaderSize(src, 1<<20) // 1 MiB buffer
	cr := csv.NewReader(br)
	cr.TrimLeadingSpace = true
	cr.ReuseRecord = true
	cr.FieldsPerRecord = 12 // we expect exactly 12 columns

	// Discard header
	if _, err := cr.Read(); err != nil {
		return &TurnScanner{err: err}
	}
	return &TurnScanner{r: cr}
}

// Scan advances to the next record.  False = EOF or error.
func (ts *TurnScanner) Scan() bool {
	if ts.err != nil {
		return false
	}

	rec, err := ts.r.Read()
	if err != nil {
		if err != io.EOF {
			ts.err = err
		}
		return false
	}

	t, perr := parseRecord(rec)
	if perr != nil {
		ts.err = perr
		return false
	}
	ts.curr = t
	return true
}

func (ts *TurnScanner) Turn() Turn { return ts.curr }
func (ts *TurnScanner) Err() error { return ts.err }

// -------------------------------------------------------------------
// CSV → Turn conversion
// -------------------------------------------------------------------
func parseRecord(f []string) (Turn, error) {
	tn, _ := strconv.Atoi(f[2])
	sc, _ := strconv.Atoi(f[5])
	tot, _ := strconv.Atoi(f[6])
	tp, _ := strconv.Atoi(f[7])
	eq, _ := strconv.ParseFloat(f[9], 64)
	tr, _ := strconv.Atoi(f[10])
	opp, _ := strconv.Atoi(f[11])

	return Turn{
		PlayerID:       f[0],
		GameID:         f[1],
		TurnNumber:     tn,
		Rack:           strings.TrimSpace(f[3]),
		Play:           strings.TrimSpace(f[4]),
		Score:          sc,
		TotalScore:     tot,
		TilesPlayed:    tp,
		Leave:          strings.TrimSpace(f[8]),
		Equity:         eq,
		TilesRemaining: tr,
		OppScore:       opp,
	}, nil
}
