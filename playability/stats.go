package playability

import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
)

// NumPhases is the number of bag-count buckets regrets are split by.
const NumPhases = 4

// Phase buckets the bag count before the move: 0 = empty bag, 1 = 1-20,
// 2 = 21-50, 3 = more than 50.
func Phase(bag int) int {
	switch {
	case bag == 0:
		return 0
	case bag <= 20:
		return 1
	case bag <= 50:
		return 2
	}
	return 3
}

// Stat accumulates the regrets of one word or one alphagram.
type Stat struct {
	N       int64 // turns where the best move formed it
	NMain   int64 // ... as the main word (words only)
	Sum     float64
	Sum2    float64
	SumMain float64 // regret from turns where it was the main word
	// Per-phase regret sums; the per-phase counts of all turns are on the shard.
	PhaseSum [NumPhases]float64
	PhaseN   [NumPhases]int64
}

func (s *Stat) Add(r float64, main bool, phase int) {
	s.N++
	if main {
		s.NMain++
		s.SumMain += r
	}
	s.Sum += r
	s.Sum2 += r * r
	s.PhaseSum[phase] += r
	s.PhaseN[phase]++
}

func (s *Stat) Merge(o *Stat) {
	s.N += o.N
	s.NMain += o.NMain
	s.Sum += o.Sum
	s.Sum2 += o.Sum2
	s.SumMain += o.SumMain
	for i := range s.PhaseSum {
		s.PhaseSum[i] += o.PhaseSum[i]
		s.PhaseN[i] += o.PhaseN[i]
	}
}

// Shard is the output of one fixed block of self-play games. Shards are
// written atomically, so a shard file either exists complete or not at all;
// that is what makes a run resumable.
type Shard struct {
	Index  int
	Games  int64
	Turns  int64
	PhaseN [NumPhases]int64 // turns per phase
	Words  map[Key]*Stat
	Alphas map[Key]*Stat
}

func NewShard(idx int) *Shard {
	return &Shard{Index: idx, Words: map[Key]*Stat{}, Alphas: map[Key]*Stat{}}
}

func (s *Shard) Merge(o *Shard) {
	s.Games += o.Games
	s.Turns += o.Turns
	for i := range s.PhaseN {
		s.PhaseN[i] += o.PhaseN[i]
	}
	merge := func(dst, src map[Key]*Stat) {
		for k, v := range src {
			if d, ok := dst[k]; ok {
				d.Merge(v)
			} else {
				c := *v
				dst[k] = &c
			}
		}
	}
	merge(s.Words, o.Words)
	merge(s.Alphas, o.Alphas)
}

func ShardPath(dir string, idx int) string {
	return filepath.Join(dir, fmt.Sprintf("shard-%06d.gob.gz", idx))
}

// WriteAtomic gob-encodes v, gzipped, to path via a temp file and rename.
func WriteAtomic(path string, v any) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	zw := gzip.NewWriter(f)
	if err := gob.NewEncoder(zw).Encode(v); err != nil {
		f.Close()
		return err
	}
	if err := zw.Close(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ReadGob reads a file written by WriteAtomic.
func ReadGob(path string, v any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	return gob.NewDecoder(zr).Decode(v)
}
