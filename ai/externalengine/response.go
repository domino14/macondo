package externalengine

// EngineResponse is the JSON structure emitted by external engines on stdout.
// Only Action is required; other fields are conditionally present.
type EngineResponse struct {
	Player          string  `json:"player"`
	Rack            string  `json:"rack"`
	CumulativeScore int     `json:"cumulative_score"`
	Action          string  `json:"action"`   // "place" | "exchange" | "pass"
	Position        string  `json:"position"` // place only, e.g. "N1"
	Tiles           string  `json:"tiles"`    // place: word (lowercase = blank); exchange: tiles
	Score           int     `json:"score"`
	Equity          float64 `json:"equity"`
	Win             float64 `json:"win"`
}
