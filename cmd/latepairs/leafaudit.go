package main

import (
	"context"
	"time"

	"github.com/domino14/macondo/ai/bot"
	"github.com/domino14/macondo/endgame/negamax"
	"github.com/domino14/macondo/game"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/move"
)

// -leafaudit K checks, without any sim, how well static play judges the
// endgames a late-game sim reaches: from each position it plays the top K
// static candidates, then static moves for both sides until the bag empties,
// and compares that endgame's static playout with an exact solve (and the
// quick 2-ply solve the sim's endgame leaves use). One leaf per candidate, so
// a position's leaves also show whether static ranks its candidates' endgames
// as the solver does.
type leafRecord struct {
	Key       string  `json:"key"`
	Bag       int     `json:"bag"` // at the position
	Cand      int     `json:"cand"`
	Play      string  `json:"play"`
	Equity    float64 `json:"equity"`
	Err       string  `json:"err,omitempty"`
	Leaf      bool    `json:"leaf"`      // reached an empty bag with both racks holding tiles
	MoverLeaf bool    `json:"moverLeaf"` // the position's mover is on turn at the leaf
	OnRack    int     `json:"onRack"`
	OppRack   int     `json:"oppRack"`
	Spread    int     `json:"spread"` // on-turn player's spread at the leaf
	// Final spreads for the player on turn at the leaf.
	ExactDone   bool    `json:"exactDone"`
	Exact       int     `json:"exact"`
	SolverFirst string  `json:"solverFirst"`
	StaticFinal int     `json:"staticFinal"`
	StaticFirst string  `json:"staticFirst"`
	QuickDone   bool    `json:"quickDone"`
	Quick       int     `json:"quick"` // sign only: the quick solve searches a (-1, 1) window
	ExactSecs   float64 `json:"exactSecs"`
}

var leafSecs = 20 * time.Second

func leafAudit(rules *game.GameRules, conf bot.BotConfig, p position, k int, tt *negamax.TranspositionTable) []leafRecord {
	key := p.src + "|" + p.gid + "|" + p.first
	g, me, err := rebuild(rules, p, p.i)
	if err != nil {
		return []leafRecord{{Key: key, Err: err.Error()}}
	}
	hb, err := bot.NewBotTurnPlayerFromGame(g.Game, &conf, pb.BotRequest_HASTY_BOT)
	if err != nil {
		return []leafRecord{{Key: key, Err: err.Error()}}
	}
	cands := hb.GenerateMoves(k)
	g.ReleaseMagpie()
	var out []leafRecord
	for ci, c := range cands {
		rec := leafRecord{Key: key, Bag: p.bag, Cand: ci, Play: c.ShortDescription(), Equity: c.Equity()}
		g, _, err := rebuild(rules, p, p.i)
		if err != nil {
			rec.Err = err.Error()
			out = append(out, rec)
			continue
		}
		auditOne(g.Game, conf, me, c, &rec, tt)
		g.ReleaseMagpie()
		out = append(out, rec)
	}
	return out
}

func auditOne(g *game.Game, conf bot.BotConfig, me int, cand *move.Move, rec *leafRecord, tt *negamax.TranspositionTable) {
	hb, err := bot.NewBotTurnPlayerFromGame(g, &conf, pb.BotRequest_HASTY_BOT)
	if err != nil {
		rec.Err = err.Error()
		return
	}
	// Play the candidate (the position is rebuilt identically), then static
	// moves until the bag empties.
	if err := g.PlayMove(cand, false, 0); err != nil {
		rec.Err = err.Error()
		return
	}
	for g.Playing() == pb.PlayState_PLAYING && g.Bag().TilesRemaining() > 0 {
		if err := g.PlayMove(hb.GenerateMoves(1)[0], false, 0); err != nil {
			rec.Err = err.Error()
			return
		}
	}
	onTurn := g.PlayerOnTurn()
	if g.Playing() != pb.PlayState_PLAYING || g.RackFor(0).NumTiles() == 0 || g.RackFor(1).NumTiles() == 0 {
		return
	}
	rec.Leaf = true
	rec.MoverLeaf = onTurn == me
	rec.OnRack, rec.OppRack = int(g.RackFor(onTurn).NumTiles()), int(g.RackFor(1-onTurn).NumTiles())
	rec.Spread = g.CurrentSpread()

	// Exact: full-window iterative deepening up to 8 plies.
	gs := g.Copy()
	gs.SetBackupMode(game.SimulationMode)
	es := &negamax.Solver{}
	if err := es.Init(gs); err != nil {
		rec.Err = err.Error()
		return
	}
	es.SetTTable(tt)
	es.SetTTableFraction(0.003)
	es.SetThreads(1)
	es.SetIterativeDeepening(true)
	ctx, cancel := context.WithTimeout(context.Background(), leafSecs)
	t0 := time.Now()
	v, seq, err := es.Solve(ctx, 8)
	rec.ExactSecs = time.Since(t0).Seconds()
	rec.ExactDone = err == nil && ctx.Err() == nil
	cancel()
	if err == nil {
		rec.Exact = int(v) + rec.Spread
		if len(seq) > 0 {
			rec.SolverFirst = seq[0].ShortDescription()
		}
	}
	gs.ReleaseMagpie()

	// Quick: what the sim's endgame leaves use.
	gq := g.Copy()
	gq.SetBackupMode(game.SimulationMode)
	gq.SetStateStackLength(1 + 2 + negamax.MaxGreedyPlayoutPlies + 5)
	qs := &negamax.Solver{}
	if err := qs.Init(gq); err == nil {
		qs.SetTTable(tt)
		qs.SetThreads(1)
		qs.SetFirstWinOptim(true)
		qs.SetSkipMaterialize(true)
		qs.SetPrunedKWGOptim(false)
		tt.Reset(0.003, gq.Board().Dim())
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		gq.SetEndgameMode(true)
		qv, _, err := qs.QuickAndDirtySolve(ctx, 2, 0)
		rec.QuickDone = err == nil && ctx.Err() == nil
		cancel()
		if err == nil {
			rec.Quick = int(qv) + rec.Spread
		}
	}
	gq.ReleaseMagpie()

	// Static playout to the end, as the plain sim's rollouts play it.
	for ply := 0; ply < 30 && g.Playing() == pb.PlayState_PLAYING; ply++ {
		m := hb.GenerateMoves(1)[0]
		if ply == 0 {
			rec.StaticFirst = m.ShortDescription()
		}
		if err := g.PlayMove(m, false, 0); err != nil {
			rec.Err = err.Error()
			return
		}
	}
	rec.StaticFinal = g.SpreadFor(onTurn)
}
