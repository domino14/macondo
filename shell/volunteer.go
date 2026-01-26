package shell

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/rs/zerolog/log"
	"google.golang.org/protobuf/proto"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gameanalysis"
	"github.com/domino14/macondo/worker"
)

const (
	VolunteerPollInterval      = 30 * time.Second
	VolunteerHeartbeatInterval = 30 * time.Second
	WooglesBaseURL             = "https://woogles.io"
)

// volunteer handles the volunteer command
func (sc *ShellController) volunteer(cmd *shellcmd) (*Response, error) {
	// Check for stop subcommand
	if len(cmd.args) > 0 && cmd.args[0] == "stop" {
		return sc.stopVolunteer()
	}

	// Start volunteer mode
	return sc.startVolunteer()
}

// startVolunteer initiates volunteer mode
func (sc *ShellController) startVolunteer() (*Response, error) {
	// Check if already in volunteer mode
	if sc.volunteerMode {
		return msg("Already in volunteer mode. Use 'volunteer stop' to exit."), nil
	}

	// Require unload first (no active game)
	if sc.game != nil {
		return msg("Please unload the current game first with 'unload' before volunteering."), nil
	}

	// Get API key from config
	apiKey := sc.config.GetString(config.ConfigWooglesApiKey)
	if apiKey == "" {
		return msg("No Woogles API key configured. Set it with: setconfig woogles-api-key YOUR_KEY"), nil
	}

	// Set volunteer mode flags
	sc.volunteerMode = true
	sc.volunteerStop = false

	// Create context for volunteer mode
	sc.volunteerCtx, sc.volunteerCancel = context.WithCancel(context.Background())

	// Start volunteer loop in background
	go sc.volunteerLoop()

	return msg("Volunteer mode started. Polling for jobs every 30 seconds...\nUse 'volunteer stop' to exit gracefully."), nil
}

// stopVolunteer signals volunteer mode to stop after current job
func (sc *ShellController) stopVolunteer() (*Response, error) {
	if !sc.volunteerMode {
		return msg("Not in volunteer mode."), nil
	}

	sc.volunteerStop = true
	return msg("Volunteer mode will stop after current job completes."), nil
}

// volunteerLoop polls for jobs and processes them
func (sc *ShellController) volunteerLoop() {
	// Get API key and create client
	apiKey := sc.config.GetString(config.ConfigWooglesApiKey)
	client := worker.NewWooglesClient(WooglesBaseURL, apiKey)

	ticker := time.NewTicker(VolunteerPollInterval)
	defer ticker.Stop()

	log.Info().Msg("volunteer loop started")

	for {
		select {
		case <-sc.volunteerCtx.Done():
			log.Info().Msg("volunteer loop cancelled")
			sc.cleanupVolunteerMode()
			return

		case <-ticker.C:
			// Check if we should stop
			if sc.volunteerStop {
				log.Info().Msg("volunteer stop requested")
				sc.cleanupVolunteerMode()
				return
			}

			// Try to claim a job
			job, err := client.ClaimJob(sc.volunteerCtx)
			if err != nil {
				log.Warn().Err(err).Msg("failed to claim job")
				writeln(fmt.Sprintf("Warning: Failed to claim job: %v", err), sc.l.Stdout())
				continue
			}

			if job == nil {
				// No jobs available
				log.Debug().Msg("no jobs available")
				continue
			}

			// Got a job! Process it
			log.Info().
				Str("job-id", job.JobID).
				Str("game-id", job.GameID).
				Msg("claimed job")
			writeln(fmt.Sprintf("Claimed job %s for game %s", job.JobID, job.GameID), sc.l.Stdout())

			// Process the job
			if err := sc.processVolunteerJob(client, job); err != nil {
				log.Error().
					Err(err).
					Str("job-id", job.JobID).
					Msg("failed to process job")
				writeln(fmt.Sprintf("Error processing job: %v", err), sc.l.Stdout())
			} else {
				log.Info().
					Str("job-id", job.JobID).
					Msg("job completed successfully")
				writeln(fmt.Sprintf("Job %s completed successfully", job.JobID), sc.l.Stdout())
			}

			// After processing, check if we should stop
			if sc.volunteerStop {
				log.Info().Msg("volunteer stop requested after job completion")
				sc.cleanupVolunteerMode()
				return
			}
		}
	}
}

// processVolunteerJob analyzes a game and submits the result
func (sc *ShellController) processVolunteerJob(client *worker.WooglesClient, job *worker.Job) error {
	ctx := sc.volunteerCtx

	// Start heartbeat ticker
	heartbeatTicker := time.NewTicker(VolunteerHeartbeatInterval)
	defer heartbeatTicker.Stop()

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
				progress := &worker.HeartbeatProgress{
					Status: "analyzing",
				}
				if err := client.SendHeartbeat(ctx, job.JobID, progress); err != nil {
					log.Warn().
						Err(err).
						Str("job-id", job.JobID).
						Msg("heartbeat failed")
				}
			}
		}
	}()

	// Fetch game history
	log.Info().
		Str("job-id", job.JobID).
		Str("game-id", job.GameID).
		Msg("fetching game history")

	history, err := client.FetchGameHistory(ctx, job.GameID)
	if err != nil {
		return fmt.Errorf("failed to fetch game history: %w", err)
	}

	// Determine lexicon from game history
	lexiconName := "NWL23" // default
	if history.Lexicon != "" {
		lexiconName = strings.ToUpper(history.Lexicon)
	}

	log.Info().
		Str("job-id", job.JobID).
		Str("game-id", job.GameID).
		Str("lexicon", lexiconName).
		Int("turns", len(history.Events)).
		Msg("game history fetched, loading lexicon")

	writeln(fmt.Sprintf("Analyzing game %s with lexicon %s (%d turns)", job.GameID, lexiconName, len(history.Events)), sc.l.Stdout())

	// Load lexicon if needed (this will download if not present)
	err = sc.options.SetLexicon([]string{lexiconName}, sc.config.WGLConfig())
	if err != nil {
		return fmt.Errorf("failed to load lexicon %s: %w", lexiconName, err)
	}

	// Create analyzer with job config
	analysisConfig := gameanalysis.DefaultAnalysisConfig()

	// Override with job-specific settings if provided
	if job.Config != nil {
		if job.Config.SimPlaysEarlyMid > 0 {
			analysisConfig.SimPlaysEarlyMid = job.Config.SimPlaysEarlyMid
		}
		if job.Config.SimPliesEarlyMid > 0 {
			analysisConfig.SimPliesEarlyMid = job.Config.SimPliesEarlyMid
		}
		if job.Config.SimStopEarlyMid > 0 {
			analysisConfig.SimStopEarlyMid = job.Config.SimStopEarlyMid
		}
		if job.Config.SimPlaysEarlyPreendgame > 0 {
			analysisConfig.SimPlaysEarlyPreEndgame = job.Config.SimPlaysEarlyPreendgame
		}
		if job.Config.SimPliesEarlyPreendgame > 0 {
			analysisConfig.SimPliesEarlyPreEndgame = job.Config.SimPliesEarlyPreendgame
		}
		if job.Config.SimStopEarlyPreendgame > 0 {
			analysisConfig.SimStopEarlyPreEndgame = job.Config.SimStopEarlyPreendgame
		}
		analysisConfig.PEGEarlyCutoff = job.Config.PEGEarlyCutoff
		if job.Config.Threads > 0 {
			analysisConfig.Threads = job.Config.Threads
		}
	}

	analyzer := gameanalysis.New(sc.config, analysisConfig)

	log.Info().
		Str("job-id", job.JobID).
		Str("game-id", job.GameID).
		Msg("starting analysis")

	// Analyze the game
	result, err := analyzer.AnalyzeGame(ctx, history)
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
	if err := client.SubmitResult(ctx, job.JobID, resultBytes); err != nil {
		return fmt.Errorf("failed to submit result: %w", err)
	}

	return nil
}

// cleanupVolunteerMode resets volunteer mode state
func (sc *ShellController) cleanupVolunteerMode() {
	sc.volunteerMode = false
	sc.volunteerStop = false
	if sc.volunteerCancel != nil {
		sc.volunteerCancel()
		sc.volunteerCancel = nil
	}
	writeln("Volunteer mode stopped.", sc.l.Stdout())
	log.Info().Msg("volunteer mode stopped")
}
