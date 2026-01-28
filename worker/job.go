package worker

// Job represents an analysis job received from Woogles
type Job struct {
	// Unique identifier for this job
	JobID string

	// Game ID to analyze (e.g., "ABC123")
	GameID string

	// Analysis configuration
	Config *Config
}

// Config is analysis configuration from the server
type Config struct {
	SimPlaysEarlyMid        int
	SimPliesEarlyMid        int
	SimStopEarlyMid         int
	SimPlaysEarlyPreendgame int
	SimPliesEarlyPreendgame int
	SimStopEarlyPreendgame  int
	PEGEarlyCutoff          bool
	Threads                 int
}

// HeartbeatProgress represents progress information sent in heartbeats
type HeartbeatProgress struct {
	// Current turn being analyzed (1-indexed)
	CurrentTurn int

	// Total number of turns to analyze
	TotalTurns int

	// Optional status message
	Status string
}
