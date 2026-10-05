package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/equity"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/turnplayer"
)

// Row layout of the trainer's frame cache (pytorch/training.py): the 85
// planes and the 4 spatial planes as packed bits (numpy unpackbits order,
// most significant bit first), then 72 float32 scalars, then 5 float32
// targets. Candidate rows carry no targets or spatial planes (zeros); the
// group carries the sim labels.
const (
	nSpatial    = 4 * 225
	packedBytes = (game.NN_N_PLANES + nSpatial + 7) / 8
	nTargets    = 5
	rowBytes    = packedBytes + game.NN_N_SCAL*4 + nTargets*4
)

// framesMain writes each labelled position as a group record for the
// trainer's ranking loss, to stdout (or -out):
//
//	uint32 payload bytes, then the payload:
//	"SGRP", uint16 n, uint16 row bytes, int16 mover's result (-1/0/1), int16 0,
//	float32 sim win[n], float32 sim equity[n], float32 win SE[n], float32 iterations[n],
//	n rows of the frame cache layout (the candidates' net inputs).
//
// Positions where the sim's win probabilities span less than -min-win-span
// (decided games: every candidate ~0 or ~1) are skipped.
func framesMain(args []string) {
	fs := flag.NewFlagSet("frames", flag.ExitOnError)
	var turnsPath, tag, posPath, labPath, lexicon, outPath string
	var minSpan float64
	fs.StringVar(&turnsPath, "turns", "", "the turn log the positions came from")
	fs.StringVar(&tag, "tag", "", "the tag used by select")
	fs.StringVar(&posPath, "positions", "", "positions file")
	fs.StringVar(&labPath, "labels", "", "labels file")
	fs.StringVar(&lexicon, "lexicon", "NWL23", "lexicon")
	fs.StringVar(&outPath, "out", "-", "output (- = stdout)")
	fs.Float64Var(&minSpan, "min-win-span", 0.005, "skip positions whose candidates' sim win probabilities span less than this")
	fs.Parse(args)
	zerolog.SetGlobalLevel(zerolog.WarnLevel)
	cfg := config.DefaultConfig()

	labels := map[string]*Label{}
	readJSONL(labPath, func(b []byte) {
		var l Label
		if json.Unmarshal(b, &l) == nil && len(l.Cands) > 1 {
			labels[l.Key] = &l
		}
	})
	target := map[string]int{}
	results := map[string]int{}
	readJSONL(posPath, func(b []byte) {
		var p Position
		if json.Unmarshal(b, &p) == nil && labels[p.Key] != nil {
			target[p.Game] = p.Turn
			results[p.Key] = p.Result
		}
	})
	leaves, err := equity.NewExhaustiveLeaveCalculator(lexicon, cfg, "")
	if err != nil {
		log.Fatal().Err(err).Msg("leaves")
	}
	out := os.Stdout
	if outPath != "-" {
		if out, err = os.Create(outPath); err != nil {
			log.Fatal().Err(err).Msg("out")
		}
		defer out.Close()
	}
	w := bufio.NewWriterSize(out, 1<<22)
	defer w.Flush()
	var written, decided int
	walkLabelled(cfg, lexicon, tag, turnsPath, target, labels, func(tp *turnplayer.BaseTurnPlayer, onturn int, hist []*move.Move, l *Label) {
		lo, hi := 1.0, 0.0
		for _, c := range l.Cands {
			lo, hi = math.Min(lo, c.Win), math.Max(hi, c.Win)
		}
		if hi-lo < minSpan {
			decided++
			return
		}
		moves, idx := parseCands(tp, onturn, l)
		if len(moves) < 2 {
			return
		}
		planes, scalars, err := tp.Game.MLVectorsForMoves(moves, leaves, hist)
		if err != nil {
			log.Fatal().Err(err).Msg("vectors")
		}
		n := len(moves)
		buf := make([]byte, 0, 12+16*n+n*rowBytes)
		buf = append(buf, "SGRP"...)
		buf = binary.LittleEndian.AppendUint16(buf, uint16(n))
		buf = binary.LittleEndian.AppendUint16(buf, uint16(rowBytes))
		buf = binary.LittleEndian.AppendUint16(buf, uint16(int16(results[l.Key])))
		buf = binary.LittleEndian.AppendUint16(buf, 0)
		for _, get := range []func(Candidate) float64{
			func(c Candidate) float64 { return c.Win },
			func(c Candidate) float64 { return c.Eq },
			func(c Candidate) float64 { return c.WinSE },
			func(c Candidate) float64 { return float64(c.Iters) },
		} {
			for _, i := range idx {
				buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(float32(get(l.Cands[i]))))
			}
		}
		for k := 0; k < n; k++ {
			row := make([]byte, rowBytes)
			pl := planes[k*game.NN_N_PLANES : (k+1)*game.NN_N_PLANES]
			for i, v := range pl {
				if v != 0 {
					row[i>>3] |= 0x80 >> (i & 7)
				}
			}
			for i, v := range scalars[k*game.NN_N_SCAL : (k+1)*game.NN_N_SCAL] {
				binary.LittleEndian.PutUint32(row[packedBytes+4*i:], math.Float32bits(v))
			}
			buf = append(buf, row...)
		}
		var hdr [4]byte
		binary.LittleEndian.PutUint32(hdr[:], uint32(len(buf)))
		w.Write(hdr[:])
		w.Write(buf)
		written++
	})
	fmt.Fprintf(os.Stderr, "%d groups written, %d decided positions skipped\n", written, decided)
}
