package worker

import (
	"context"
	"fmt"
	"time"

	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/proto"

	"github.com/domino14/macondo/gameanalysis"
)

// AnalysisWorker polls for analysis jobs and processes them
type AnalysisWorker struct {
	config   *WorkerConfig
	client   *WooglesClient
	analyzer *gameanalysis.Analyzer
}

// NewAnalysisWorker creates a new worker
func NewAnalysisWorker(cfg *WorkerConfig) *AnalysisWorker {
	client := NewWooglesClient(cfg.WooglesBaseURL, cfg.APIKey)

	// Create analyzer with default config if none provided
	analysisConfig := gameanalysis.DefaultAnalysisConfig()

	analyzer := gameanalysis.New(cfg.MacondoConfig, analysisConfig)

	return &AnalysisWorker{
		config:   cfg,
		client:   client,
		analyzer: analyzer,
	}
}

// Run starts the worker main loop
func (w *AnalysisWorker) Run(ctx context.Context) error {
	log.Info().
		Str("woogles-url", w.config.WooglesBaseURL).
		Dur("poll-interval", w.config.PollInterval).
		Dur("heartbeat-interval", w.config.HeartbeatInterval).
		Msg("starting analysis worker")

	pollTicker := time.NewTicker(w.config.PollInterval)
	defer pollTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Info().Msg("worker shutting down")
			return ctx.Err()

		case <-pollTicker.C:
			// Try to claim a job
			job, err := w.client.ClaimJob(ctx)
			if err != nil {
				log.Warn().Err(err).Msg("failed to claim job")
				continue
			}

			if job == nil {
				// No jobs available, continue polling
				log.Debug().Msg("no jobs available")
				continue
			}

			// Process the job
			log.Info().
				Str("job-id", job.JobID).
				Str("game-id", job.GameID).
				Msg("claimed job")

			if err := w.processJob(ctx, job); err != nil {
				log.Error().
					Err(err).
					Str("job-id", job.JobID).
					Msg("failed to process job")
			}
		}
	}
}

// processJob analyzes a game and submits the results
func (w *AnalysisWorker) processJob(ctx context.Context, job *Job) error {
	// Create a context with heartbeat ticker
	heartbeatTicker := time.NewTicker(w.config.HeartbeatInterval)
	defer heartbeatTicker.Stop()

	// Channel to signal completion
	done := make(chan struct{})
	defer close(done)

	// Start heartbeat goroutine
	go func() {
		for {
			select {
			case <-done:
				return
			case <-ctx.Done():
				return
			case <-heartbeatTicker.C:
				progress := &HeartbeatProgress{
					Status: "analyzing",
				}
				if err := w.client.SendHeartbeat(ctx, job.JobID, progress); err != nil {
					log.Warn().
						Err(err).
						Str("job-id", job.JobID).
						Msg("heartbeat failed")
					// If heartbeat fails, the job may have been reclaimed
					return
				}
				log.Debug().Str("job-id", job.JobID).Msg("sent heartbeat")
			}
		}
	}()

	// Fetch the game history
	log.Info().
		Str("job-id", job.JobID).
		Str("game-id", job.GameID).
		Msg("fetching game history")

	history, err := w.client.FetchGameHistory(ctx, job.GameID)
	if err != nil {
		return fmt.Errorf("failed to fetch game history: %w", err)
	}

	log.Info().
		Str("job-id", job.JobID).
		Str("game-id", job.GameID).
		Int("turns", len(history.Events)).
		Msg("game history fetched, starting analysis")

	// Analyze the game
	result, err := w.analyzer.AnalyzeGame(ctx, history)
	if err != nil {
		return fmt.Errorf("failed to analyze game: %w", err)
	}

	log.Info().
		Str("job-id", job.JobID).
		Str("game-id", job.GameID).
		Int("turns-analyzed", len(result.Turns)).
		Msg("analysis complete")

	// Convert result to protobuf
	resultProto := result.ToProto()

	// Serialize to bytes
	resultBytes, err := proto.Marshal(resultProto)
	if err != nil {
		return fmt.Errorf("failed to marshal result: %w", err)
	}

	log.Info().
		Str("job-id", job.JobID).
		Int("result-size", len(resultBytes)).
		Msg("submitting result")

	// Submit the result
	if err := w.client.SubmitResult(ctx, job.JobID, resultBytes); err != nil {
		return fmt.Errorf("failed to submit result: %w", err)
	}

	log.Info().
		Str("job-id", job.JobID).
		Str("game-id", job.GameID).
		Msg("job completed successfully")

	return nil
}
