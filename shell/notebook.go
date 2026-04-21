package shell

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/domino14/word-golib/tilemapping"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

// NotebookOutput is structured output from a notebook cell command execution.
type NotebookOutput struct {
	Kind string      `json:"kind"`
	Data interface{} `json:"data"`
}

// PlayerInfo holds name and score for one player.
type PlayerInfo struct {
	Name  string `json:"name"`
	Score int    `json:"score"`
}

// BoardData is a full snapshot of the board state for the notebook frontend.
type BoardData struct {
	FEN            string         `json:"fen"`
	Rack           string         `json:"rack"`
	PlayerOnTurn   int            `json:"playerOnTurn"`
	Players        []PlayerInfo   `json:"players"`
	BagCount       int            `json:"bagCount"`
	TurnNumber     int            `json:"turnNumber"`
	LastPlay       string         `json:"lastPlay,omitempty"`
	BonusLayout    []string       `json:"bonusLayout"`
	Dimension      int            `json:"dimension"`
	AlphabetScores map[string]int `json:"alphabetScores"`
	RemainingTiles map[string]int `json:"remainingTiles"`
}

// MoveRow is one entry in a generated move list.
type MoveRow struct {
	Rank   int     `json:"rank"`
	Move   string  `json:"move"`
	Leave  string  `json:"leave"`
	Score  int     `json:"score"`
	Equity float64 `json:"equity"`
}

// MoveTableData is the structured move list returned by the gen command.
type MoveTableData struct {
	Moves []MoveRow `json:"moves"`
}

// getBoardData extracts structured board state from the current game.
func (sc *ShellController) getBoardData() (*BoardData, error) {
	if sc.game == nil {
		return nil, errors.New("no game loaded")
	}

	bd := sc.game.Board()
	alph := sc.game.Alphabet()
	dim := bd.Dim()

	fen := bd.ToFEN(alph)

	rack := ""
	if r := sc.game.RackFor(sc.game.PlayerOnTurn()); r != nil {
		rack = r.String()
	}

	// Player info
	history := sc.game.History()
	players := make([]PlayerInfo, sc.game.NumPlayers())
	for i := 0; i < sc.game.NumPlayers(); i++ {
		name := fmt.Sprintf("Player %d", i+1)
		if history != nil && len(history.Players) > i {
			if n := history.Players[i].Nickname; n != "" {
				name = n
			}
		}
		players[i] = PlayerInfo{Name: name, Score: sc.game.PointsFor(i)}
	}

	// Bag + remaining tiles (bag + opponent rack)
	bag := sc.game.Bag()
	remainingTiles := make(map[string]int)
	for _, tile := range bag.Tiles() {
		letter := tile.UserVisible(alph, false)
		remainingTiles[letter]++
	}
	opp := (sc.game.PlayerOnTurn() + 1) % sc.game.NumPlayers()
	if oppRack := sc.game.RackFor(opp); oppRack != nil {
		for _, tile := range oppRack.TilesOn() {
			letter := tile.UserVisible(alph, false)
			remainingTiles[letter]++
		}
	}

	// Alphabet scores
	dist := bag.LetterDistribution()
	alphabetScores := make(map[string]int)
	for i := tilemapping.MachineLetter(0); i < tilemapping.MachineLetter(alph.NumLetters()); i++ {
		letter := i.UserVisible(alph, false)
		cleanLetter := strings.ReplaceAll(strings.ReplaceAll(letter, "[", ""), "]", "")
		alphabetScores[cleanLetter] = int(dist.Score(i))
	}
	alphabetScores["?"] = 0

	// Bonus layout (raw bonus bytes per row, same as render3D)
	bonusLayout := make([]string, dim)
	for row := 0; row < dim; row++ {
		rowBytes := make([]byte, dim)
		for col := 0; col < dim; col++ {
			rowBytes[col] = byte(bd.GetBonus(row, col))
		}
		bonusLayout[row] = string(rowBytes)
	}

	// Last play summary
	lastPlay := ""
	if history != nil && sc.curTurnNum > 0 && len(history.Events) >= sc.curTurnNum {
		evt := history.Events[sc.curTurnNum-1]
		who := fmt.Sprintf("Player %d", evt.PlayerIndex+1)
		if len(history.Players) > int(evt.PlayerIndex) {
			who = history.Players[evt.PlayerIndex].Nickname
		}
		switch evt.Type {
		case pb.GameEvent_TILE_PLACEMENT_MOVE:
			lastPlay = fmt.Sprintf("%s played %s %s for %d pts", who, evt.Position, evt.PlayedTiles, evt.Score)
		case pb.GameEvent_PASS:
			lastPlay = fmt.Sprintf("%s passed", who)
		case pb.GameEvent_EXCHANGE:
			lastPlay = fmt.Sprintf("%s exchanged %s", who, evt.Exchanged)
		case pb.GameEvent_CHALLENGE:
			lastPlay = fmt.Sprintf("%s challenged", who)
		}
	}

	return &BoardData{
		FEN:            fen,
		Rack:           rack,
		PlayerOnTurn:   sc.game.PlayerOnTurn(),
		Players:        players,
		BagCount:       bag.TilesRemaining(),
		TurnNumber:     sc.curTurnNum,
		LastPlay:       lastPlay,
		BonusLayout:    bonusLayout,
		Dimension:      dim,
		AlphabetScores: alphabetScores,
		RemainingTiles: remainingTiles,
	}, nil
}

// getMoveTableData extracts structured move data from sc.curPlayList.
func (sc *ShellController) getMoveTableData() *MoveTableData {
	if sc.game == nil || len(sc.curPlayList) == 0 {
		return &MoveTableData{Moves: []MoveRow{}}
	}
	alph := sc.game.Alphabet()
	bd := sc.game.Board()
	rows := make([]MoveRow, len(sc.curPlayList))
	for i, m := range sc.curPlayList {
		rows[i] = MoveRow{
			Rank:   i + 1,
			Move:   bd.MoveDescriptionWithPlaythrough(m),
			Leave:  m.Leave().UserVisible(alph),
			Score:  m.Score(),
			Equity: m.Equity(),
		}
	}
	return &MoveTableData{Moves: rows}
}

// boardAffectingCmds lists commands that produce board state as output.
var boardAffectingCmds = map[string]bool{
	"load": true, "new": true, "s": true, "show": true,
	"n": true, "p": true, "turn": true, "last": true,
	"commit": true, "aiplay": true, "hastyplay": true, "rack": true,
	"challenge": true,
}

// NotebookExecute executes a single command line and returns structured outputs
// for use by the notebook HTTP server.
func (sc *ShellController) NotebookExecute(line string) ([]*NotebookOutput, error) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return nil, nil
	}

	expandedLine, err := sc.expandAliases(line)
	if err != nil {
		return nil, err
	}

	cmd, err := extractFields(expandedLine)
	if err != nil {
		if errors.Is(err, errNoData) {
			return nil, nil
		}
		return nil, err
	}

	if cmd.cmd == "exit" {
		return nil, errors.New("exit is not supported in notebook mode")
	}

	// Capture any text written via showMessage during command execution.
	var buf strings.Builder
	sc.SetWriter(&buf)
	defer sc.SetWriter(nil)

	resp, cmdErr := sc.dispatchForNotebook(cmd)

	capturedText := buf.String()
	if resp != nil && resp.message != "" {
		capturedText += resp.message
	}

	if cmdErr != nil {
		return nil, cmdErr
	}

	var outputs []*NotebookOutput

	// gen: emit only the move table (board was already shown by the preceding load/show).
	if cmd.cmd == "gen" && len(sc.curPlayList) > 0 {
		return []*NotebookOutput{{Kind: "table", Data: sc.getMoveTableData()}}, nil
	}

	// Board-affecting commands: emit only the graphic board, dropping the
	// captured ASCII text (which would be redundant).
	if boardAffectingCmds[cmd.cmd] && sc.game != nil {
		if boardData, err := sc.getBoardData(); err == nil {
			outputs = append(outputs, &NotebookOutput{Kind: "board", Data: boardData})
		}
		return outputs, nil
	}

	// All other commands: emit captured text (sim results, analysis, etc.).
	if capturedText != "" {
		outputs = append(outputs, &NotebookOutput{Kind: "text", Data: capturedText})
	}

	return outputs, nil
}

// ProgressData is sent as the data for "progress" kind outputs during long operations.
type ProgressData struct {
	Message    string `json:"message"`
	Iterations int    `json:"iterations,omitempty"`
}

// NotebookExecuteStreaming executes a single command line and sends structured
// outputs to outCh as they become available. For long-running operations
// (sim, endgame, peg) it streams progress updates until completion or ctx
// cancellation.
func (sc *ShellController) NotebookExecuteStreaming(ctx context.Context, line string, outCh chan<- *NotebookOutput) error {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return nil
	}

	expandedLine, err := sc.expandAliases(line)
	if err != nil {
		return err
	}

	cmd, err := extractFields(expandedLine)
	if err != nil {
		if errors.Is(err, errNoData) {
			return nil
		}
		return err
	}

	if cmd.cmd == "exit" {
		return errors.New("exit is not supported in notebook mode")
	}

	// Route streaming-capable long-running commands.
	switch cmd.cmd {
	case "sim":
		// Sub-commands (stop, show, details, …) fall through to regular execute.
		if len(cmd.args) == 0 {
			return sc.streamSim(ctx, cmd, outCh)
		}
	case "endgame":
		if len(cmd.args) == 0 {
			return sc.streamEndgame(ctx, cmd, outCh)
		}
	case "peg":
		if len(cmd.args) == 0 {
			return sc.streamPeg(ctx, cmd, outCh)
		}
	}

	// Fall back to synchronous execution for everything else.
	outputs, err := sc.NotebookExecute(line)
	if err != nil {
		return err
	}
	for _, out := range outputs {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case outCh <- out:
		}
	}
	return nil
}

// streamSim runs simulation with progress streaming.
func (sc *ShellController) streamSim(ctx context.Context, cmd *shellcmd, outCh chan<- *NotebookOutput) error {
	params, err := sc.simPrepare(cmd.options)
	if err != nil {
		return err
	}

	if params.fixediters != 0 {
		// Experiment mode: just run synchronously — no incremental progress available.
		outCh <- &NotebookOutput{Kind: "progress", Data: ProgressData{Message: "Running sim experiment..."}}
		result, err := sc.simRunSync()
		if err != nil {
			return err
		}
		outCh <- &NotebookOutput{Kind: "text", Data: result}
		return nil
	}

	// Use the parent ctx so cancellation from the notebook server propagates.
	simCtx, simCancel := context.WithCancel(ctx)
	sc.simCtx = simCtx
	sc.simCancel = simCancel
	defer simCancel()

	type simResult struct {
		err error
	}
	doneCh := make(chan simResult, 1)
	go func() {
		doneCh <- simResult{err: sc.simmer.Simulate(simCtx)}
	}()

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	outCh <- &NotebookOutput{Kind: "progress", Data: ProgressData{Message: "Simulation started..."}}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case res := <-doneCh:
			ticker.Stop()
			if res.err != nil && !errors.Is(res.err, context.Canceled) {
				return res.err
			}
			winning := sc.simmer.WinningPlay()
			var result string
			if winning != nil {
				result = "Sim winner: " + winning.Move().ShortDescription() + "\n"
			}
			result += sc.simmer.EquityStats()
			outCh <- &NotebookOutput{Kind: "text", Data: result}
			if sc.game != nil {
				if bd, bdErr := sc.getBoardData(); bdErr == nil {
					outCh <- &NotebookOutput{Kind: "board", Data: bd}
				}
			}
			return nil
		case <-ticker.C:
			iters := sc.simmer.Iterations()
			outCh <- &NotebookOutput{Kind: "progress", Data: ProgressData{
				Message:    fmt.Sprintf("Simulating... %d iterations", iters),
				Iterations: iters,
			}}
		}
	}
}

// streamEndgame runs the endgame solver with progress streaming.
func (sc *ShellController) streamEndgame(ctx context.Context, cmd *shellcmd, outCh chan<- *NotebookOutput) error {
	params, err := sc.endgamePrepare(cmd) // sets sc.endgameCtx, sc.endgameCancel
	if err != nil {
		return err
	}

	// Wire parent ctx → endgame cancel so POST /api/cancel propagates.
	go func() {
		select {
		case <-ctx.Done():
			sc.endgameCancel()
		case <-sc.endgameCtx.Done():
		}
	}()

	type endResult struct {
		result string
		err    error
	}
	doneCh := make(chan endResult, 1)
	go func() {
		res, err := sc.endgameRunSync(params)
		doneCh <- endResult{res, err}
	}()

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	outCh <- &NotebookOutput{Kind: "progress", Data: ProgressData{Message: "Endgame solver started..."}}

	for {
		select {
		case <-ctx.Done():
			sc.endgameCancel()
			return ctx.Err()
		case res := <-doneCh:
			ticker.Stop()
			if res.err != nil && !errors.Is(res.err, context.Canceled) {
				return res.err
			}
			if res.result != "" {
				outCh <- &NotebookOutput{Kind: "text", Data: res.result}
			}
			if sc.game != nil {
				if bd, bdErr := sc.getBoardData(); bdErr == nil {
					outCh <- &NotebookOutput{Kind: "board", Data: bd}
				}
			}
			return nil
		case <-ticker.C:
			outCh <- &NotebookOutput{Kind: "progress", Data: ProgressData{Message: "Solving endgame..."}}
		}
	}
}

// streamPeg runs the pre-endgame solver with progress streaming.
func (sc *ShellController) streamPeg(ctx context.Context, cmd *shellcmd, outCh chan<- *NotebookOutput) error {
	params, err := sc.pegPrepare(cmd) // sets sc.pegCtx, sc.pegCancel
	if err != nil {
		return err
	}

	go func() {
		select {
		case <-ctx.Done():
			sc.pegCancel()
		case <-sc.pegCtx.Done():
		}
	}()

	type pegResult struct {
		result string
		err    error
	}
	doneCh := make(chan pegResult, 1)
	go func() {
		res, err := sc.pegRunSync(params)
		doneCh <- pegResult{res, err}
	}()

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	outCh <- &NotebookOutput{Kind: "progress", Data: ProgressData{Message: "Pre-endgame solver started..."}}

	for {
		select {
		case <-ctx.Done():
			sc.pegCancel()
			return ctx.Err()
		case res := <-doneCh:
			ticker.Stop()
			if res.err != nil && !errors.Is(res.err, context.Canceled) {
				return res.err
			}
			if res.result != "" {
				outCh <- &NotebookOutput{Kind: "text", Data: res.result}
			}
			if sc.game != nil {
				if bd, bdErr := sc.getBoardData(); bdErr == nil {
					outCh <- &NotebookOutput{Kind: "board", Data: bd}
				}
			}
			return nil
		case <-ticker.C:
			outCh <- &NotebookOutput{Kind: "progress", Data: ProgressData{Message: "Solving pre-endgame..."}}
		}
	}
}

// dispatchForNotebook dispatches a parsed command to the appropriate handler.
// It mirrors standardModeSwitch but excludes commands that are incompatible
// with the notebook (exit, autoplay, volunteer, etc.).
func (sc *ShellController) dispatchForNotebook(cmd *shellcmd) (*Response, error) {
	switch cmd.cmd {
	case "help":
		return sc.help(cmd)
	case "alias":
		return sc.alias(cmd)
	case "new":
		return sc.newGame(cmd)
	case "load":
		return sc.load(cmd)
	case "unload":
		return sc.unload(cmd)
	case "last":
		return sc.last(cmd)
	case "n":
		return sc.next(cmd)
	case "p":
		return sc.prev(cmd)
	case "s", "show":
		return sc.show(cmd)
	case "name":
		return sc.name(cmd)
	case "note":
		return sc.note(cmd)
	case "turn":
		return sc.turn(cmd)
	case "rack":
		return sc.rack(cmd)
	case "set":
		return sc.set(cmd)
	case "setconfig":
		return sc.setConfig(cmd)
	case "gen":
		return sc.generate(cmd)
	case "sim":
		return sc.sim(cmd)
	case "infer":
		return sc.infer(cmd)
	case "add":
		return sc.add(cmd)
	case "challenge":
		return sc.challenge(cmd)
	case "commit":
		return sc.commit(cmd)
	case "aiplay":
		return sc.eliteplay(cmd)
	case "hastyplay":
		return sc.hastyplay(cmd)
	case "list":
		return sc.list(cmd)
	case "var", "variation", "variations":
		return sc.variation(cmd)
	case "endgame":
		return sc.endgame(cmd)
	case "peg":
		return sc.preendgame(cmd)
	case "mode":
		return sc.setMode(cmd)
	case "export":
		return sc.export(cmd)
	case "analyze":
		return sc.analyze(cmd)
	case "analyze-turn":
		return sc.analyzeTurn(cmd)
	case "analyze-batch":
		return sc.analyzeBatch(cmd)
	case "analyze-browse":
		return sc.analyzeBrowse(cmd)
	case "analyze-view":
		return sc.analyzeView(cmd)
	case "autoanalyze":
		return sc.autoAnalyze(cmd)
	case "speedtest":
		return sc.speedtest(cmd)
	case "script":
		return sc.script(cmd)
	case "gid":
		return sc.gid(cmd)
	case "leave":
		return sc.leave(cmd)
	case "cgp":
		return sc.cgp(cmd)
	case "check":
		return sc.check(cmd)
	case "gamestate":
		return sc.gameState(cmd)
	case "mleval":
		return sc.mleval(cmd)
	case "winpct":
		return sc.winpct(cmd)
	case "explain":
		return sc.explain(cmd)
	case "build-wmp":
		return sc.buildWMP(cmd)
	case "autoplay", "selftest", "volunteer", "update", "render":
		return nil, fmt.Errorf("command %q is not supported in notebook mode", cmd.cmd)
	default:
		return nil, fmt.Errorf("command %q not found", cmd.cmd)
	}
}
