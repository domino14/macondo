package worker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/domino14/macondo/config"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/gcgio"
)

// WooglesClient handles HTTP communication with the Woogles API
type WooglesClient struct {
	baseURL    string
	apiKey     string
	version    string
	httpClient *http.Client
	cfg        *config.Config
}

// NewWooglesClient creates a new Woogles API client
func NewWooglesClient(baseURL, apiKey string, cfg *config.Config, version string) *WooglesClient {
	return &WooglesClient{
		baseURL: baseURL,
		apiKey:  apiKey,
		version: version,
		httpClient: &http.Client{
			Timeout: 30 * time.Second, // Timeout for all HTTP requests
		},
		cfg: cfg,
	}
}

// ClaimJob attempts to claim a job from the queue
// Uses Connect RPC JSON format
func (c *WooglesClient) ClaimJob(ctx context.Context) (*Job, error) {
	url := c.baseURL + "/api/analysis_service.AnalysisQueueService/ClaimJob"

	reqBody, err := json.Marshal(map[string]interface{}{
		"macondoVersion": c.version,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Api-Key", c.apiKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Parse Connect RPC JSON response
	var claimResp struct {
		NoJobs bool   `json:"no_jobs"`
		JobId  string `json:"job_id"`
		GameId string `json:"game_id"`
		Config struct {
			SimPlaysEarlyMid        int32 `json:"sim_plays_early_mid"`
			SimPliesEarlyMid        int32 `json:"sim_plies_early_mid"`
			SimStopEarlyMid         int32 `json:"sim_stop_early_mid"`
			SimPlaysEarlyPreendgame int32 `json:"sim_plays_early_preendgame"`
			SimPliesEarlyPreendgame int32 `json:"sim_plies_early_preendgame"`
			SimStopEarlyPreendgame  int32 `json:"sim_stop_early_preendgame"`
			PegEarlyCutoff          bool  `json:"peg_early_cutoff"`
			Threads                 int32 `json:"threads"`
		} `json:"config"`
	}

	if err := json.Unmarshal(body, &claimResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if claimResp.NoJobs {
		return nil, nil // No jobs available
	}

	// Convert to our Job struct with gameanalysis.AnalysisConfig
	return &Job{
		JobID:  claimResp.JobId,
		GameID: claimResp.GameId,
		Config: &Config{
			SimPlaysEarlyMid:        int(claimResp.Config.SimPlaysEarlyMid),
			SimPliesEarlyMid:        int(claimResp.Config.SimPliesEarlyMid),
			SimStopEarlyMid:         int(claimResp.Config.SimStopEarlyMid),
			SimPlaysEarlyPreendgame: int(claimResp.Config.SimPlaysEarlyPreendgame),
			SimPliesEarlyPreendgame: int(claimResp.Config.SimPliesEarlyPreendgame),
			SimStopEarlyPreendgame:  int(claimResp.Config.SimStopEarlyPreendgame),
			PEGEarlyCutoff:          claimResp.Config.PegEarlyCutoff,
			Threads:                 int(claimResp.Config.Threads),
		},
	}, nil
}

// SubmitResult submits analysis results back to Woogles.
// resultJSON must be the protojson encoding of a macondo.GameAnalysisResult;
// it is embedded inline (not base64) so the server deserializes it as a typed proto.
func (c *WooglesClient) SubmitResult(ctx context.Context, jobID string, resultJSON []byte) error {
	url := c.baseURL + "/api/analysis_service.AnalysisQueueService/SubmitResult"

	// Embed resultJSON as a raw JSON value so the server receives a typed
	// macondo.GameAnalysisResult in the "result" field, not base64 bytes.
	reqBody, err := json.Marshal(map[string]interface{}{
		"jobId":          jobID,
		"macondoVersion": c.version,
		"result":         json.RawMessage(resultJSON),
	})
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Api-Key", c.apiKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	var submitResp struct {
		Accepted        bool   `json:"accepted"`
		Error           string `json:"error"`
		ExpectedVersion string `json:"expectedVersion"`
	}
	if err := json.Unmarshal(body, &submitResp); err != nil {
		return fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if !submitResp.Accepted {
		if submitResp.ExpectedVersion != "" {
			return fmt.Errorf("result rejected (version mismatch: server wants %s, we sent %s): %s",
				submitResp.ExpectedVersion, c.version, submitResp.Error)
		}
		return fmt.Errorf("result rejected: %s", submitResp.Error)
	}

	return nil
}

// SendHeartbeat sends a heartbeat to indicate the worker is still processing
func (c *WooglesClient) SendHeartbeat(ctx context.Context, jobID string, progress *HeartbeatProgress) error {
	url := c.baseURL + "/api/analysis_service.AnalysisQueueService/Heartbeat"

	req := map[string]interface{}{
		"jobId": jobID,
	}
	if progress != nil {
		req["currentTurn"] = progress.CurrentTurn
		req["totalTurns"] = progress.TotalTurns
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Api-Key", c.apiKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// 410 Gone means the job was reclaimed by the server
	if resp.StatusCode == http.StatusGone {
		return fmt.Errorf("job was reclaimed by server")
	}

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	var hbResp struct {
		Continue bool `json:"continue"`
	}
	if err := json.Unmarshal(body, &hbResp); err != nil {
		return fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if !hbResp.Continue {
		return fmt.Errorf("server requested stop")
	}

	return nil
}

// FetchGameHistory fetches the game history from Woogles
func (c *WooglesClient) FetchGameHistory(ctx context.Context, gameID string) (*pb.GameHistory, error) {
	url := c.baseURL + "/api/game_service.GameMetadataService/GetGCG"

	reqBody := fmt.Sprintf(`{"gameId": "%s"}`, gameID)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	type gcgStruct struct {
		GCG string `json:"gcg"`
	}
	var gcgObj gcgStruct

	if err := json.Unmarshal(body, &gcgObj); err != nil {
		return nil, fmt.Errorf("failed to unmarshal GCG response: %w", err)
	}

	history, err := gcgio.ParseGCGFromReader(c.cfg, strings.NewReader(gcgObj.GCG))
	if err != nil {
		return nil, fmt.Errorf("failed to parse GCG: %w", err)
	}

	return history, nil
}
