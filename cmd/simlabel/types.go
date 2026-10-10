package main

// Position is one line of select's output and sim's input.
type Position struct {
	Key    string `json:"key"`    // tag:gameID:turn, unique across logs
	Game   string `json:"game"`   // game ID in the turn log
	Turn   int    `json:"turn"`   // the turn about to be played
	CGP    string `json:"cgp"`    // the position, mover on turn, opponent's rack hidden
	Played string `json:"played"` // the move the log's bot played here
	Result int    `json:"result"` // mover's final result: 1 win, 0 draw, -1 loss
	Spread int    `json:"spread"` // mover's spread now (before the move)
	Final  int    `json:"final"`  // mover's final spread
	Unseen int    `json:"unseen"` // tiles in the bag plus the opponent's rack
}

// Candidate is one simulated play.
type Candidate struct {
	Move    string  `json:"move"`
	Score   int     `json:"score"`
	Equity  float64 `json:"static_eq"` // static equity (the ranking the sim started from)
	Win     float64 `json:"win"`       // sim win probability for the mover
	WinSE   float64 `json:"win_se"`
	Eq      float64 `json:"eq"` // sim equity: points spread change plus leftover
	EqSE    float64 `json:"eq_se"`
	Iters   int     `json:"iters"`
	Ignored bool    `json:"pruned"` // dropped by the stopping rule before the end
}

// Label is one line of sim's output.
type Label struct {
	Key        string      `json:"key"`
	Plies      int         `json:"plies"`
	Iterations int         `json:"iterations"` // the sim's total iteration count
	Seconds    float64     `json:"seconds"`
	Cands      []Candidate `json:"cands"` // best sim win probability first
}
