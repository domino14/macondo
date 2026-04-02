package gameanalysis

import "github.com/domino14/macondo/move"

// SimPlayResult holds per-play statistics from Monte Carlo simulation.
type SimPlayResult struct {
	MoveDescription string
	Score           int
	Leave           string
	IsBingo         bool
	WinProb         float64
	WinProbStdErr   float64
	Equity          float64
	EquityStdErr    float64
	Iterations      int
	PlyStats        []*PlyStatResult
	IsPlayedMove    bool
	IsIgnored       bool
}

// PlyStatResult holds per-ply score and bingo statistics for a simmed play.
type PlyStatResult struct {
	Ply        int
	ScoreMean  float64
	ScoreStdev float64
	BingoPct   float64
}

// PEGPlayResult holds per-play statistics from the pre-endgame solver.
type PEGPlayResult struct {
	MoveDescription string
	Score           int
	Leave           string
	IsBingo         bool
	WinProb         float64
	Outcomes        []*PEGOutcomeResult
	HasSpread       bool
	AvgSpread       float64
	IsPlayedMove    bool
	IsIgnored       bool
}

// PEGOutcomeResult holds a single tile-draw outcome from PEG analysis.
type PEGOutcomeResult struct {
	Tiles   string // human-readable, e.g. "A", "EI"
	Outcome string // "win", "draw", "loss", "unknown"
	Count   int
}

// EndgameVariationResult holds a move sequence from endgame analysis.
type EndgameVariationResult struct {
	Moves       []*EndgameMoveResult
	FinalSpread int16
}

// EndgameMoveResult is a single move within an endgame variation.
type EndgameMoveResult struct {
	MoveDescription string
	Score           int
	MoveNumber      int
}

// GamePhase represents the phase of the game based on tiles remaining in bag.
type GamePhase int

const (
	// PhaseEarlyMid represents early/mid game (>7 tiles in bag)
	PhaseEarlyMid GamePhase = iota
	// PhaseEarlyPreEndgame represents early pre-endgame (2-7 tiles in bag)
	PhaseEarlyPreEndgame
	// PhasePreEndgame represents pre-endgame (1 tile in bag, uses PEG solver)
	PhasePreEndgame
	// PhaseEndgame represents endgame (0 tiles in bag, uses negamax)
	PhaseEndgame
)

// String returns a string representation of the game phase.
func (p GamePhase) String() string {
	switch p {
	case PhaseEarlyMid:
		return "Early"
	case PhaseEarlyPreEndgame:
		return "EarlyPE"
	case PhasePreEndgame:
		return "PEG"
	case PhaseEndgame:
		return "Endgame"
	default:
		return "Unknown"
	}
}

// TurnAnalysis contains the analysis results for a single turn.
type TurnAnalysis struct {
	TurnNumber  int
	PlayerIndex int
	PlayerName  string
	Rack        string
	Phase       GamePhase
	TilesInBag  int // Number of tiles in bag at this turn

	// The move that was actually played
	PlayedMove *move.Move
	// The optimal move according to analysis
	OptimalMove *move.Move
	// Display strings used when loaded from storage (PlayedMove/OptimalMove are nil)
	PlayedMoveStr  string
	OptimalMoveStr string

	// For sim/PEG phases - win probability
	PlayedWinProb  float64
	OptimalWinProb float64
	WinProbLoss    float64 // OptimalWinProb - PlayedWinProb

	// For endgame phase - spread difference
	SpreadLoss         int16 // How much worse the played move is compared to optimal
	OptimalFinalSpread int16 // The final spread with the optimal move (endgame only)
	CurrentSpread      int   // The spread before this move (endgame only, for blown endgame detection)

	// Whether the played move was optimal
	WasOptimal bool

	// Mistake categorization
	MistakeCategory string // "Small", "Medium", "Large", or empty if optimal
	BlownEndgame    bool   // True if the mistake changed a win to a loss/tie in endgame

	// Bingo tracking
	OptimalIsBingo bool // True if the optimal move is a bingo (7 tiles)
	PlayedIsBingo  bool // True if the played move is a bingo (7 tiles)
	MissedBingo    bool // True if optimal was a bingo but player didn't play a bingo

	// Phony handling
	IsPhony         bool // The played move was a phony
	PhonyChallenged bool // A phony that was challenged off
	MissedChallenge bool // Player failed to challenge opponent's phony

	// KnownOppRack contains the known opponent tiles (if any) used in analysis.
	// This is populated when the opponent's phony was challenged off, exposing their rack.
	KnownOppRack string

	// Enriched data (v2+)
	TopSimPlays        []*SimPlayResult
	TopPEGPlays        []*PEGPlayResult
	PrincipalVariation *EndgameVariationResult
	OtherVariations    []*EndgameVariationResult
}

// PlayerSummary contains aggregate statistics for a player across the game.
type PlayerSummary struct {
	PlayerName     string
	TurnsPlayed    int
	OptimalMoves   int
	AvgWinProbLoss float64

	// Mistake breakdown
	SmallMistakes  int
	MediumMistakes int
	LargeMistakes  int

	MistakeIndex float64 // Sum of mistake points (0.2 for small, 0.5 for medium, 1.0 for large)
	EstimatedELO float64 // Estimated ELO based on mistake index

	// Bingo tracking
	AvailableBingos int // Number of turns where optimal move was a bingo
	MissedBingos    int // Number of times player didn't play a bingo when optimal was a bingo
}

// GameAnalysisResult contains the complete analysis results for a game.
type GameAnalysisResult struct {
	Turns           []*TurnAnalysis
	PlayerSummaries [2]*PlayerSummary
	AnalysisVersion int
	AnalyzerVersion string
}
