package worker

import (
	"os"
	"time"

	"github.com/domino14/macondo/config"
)

// WorkerConfig holds configuration for the analysis worker
type WorkerConfig struct {
	// Base URL for the Woogles API (default: https://woogles.io)
	WooglesBaseURL string

	// Woogles API key for authentication
	APIKey string

	// How often to poll for new jobs when idle
	PollInterval time.Duration

	// How often to send heartbeats while processing
	HeartbeatInterval time.Duration

	// Macondo configuration for the analyzer
	MacondoConfig *config.Config
}

// DefaultWorkerConfig creates a WorkerConfig with default values
func DefaultWorkerConfig() *WorkerConfig {
	return &WorkerConfig{
		WooglesBaseURL:    getEnv("MACONDO_WOOGLES_URL", "https://woogles.io"),
		APIKey:            getEnv("MACONDO_WOOGLES_API_KEY", ""),
		PollInterval:      getEnvDuration("MACONDO_WORKER_POLL_INTERVAL", 5*time.Second),
		HeartbeatInterval: getEnvDuration("MACONDO_WORKER_HEARTBEAT_INTERVAL", 30*time.Second),
		MacondoConfig:     config.DefaultConfig(),
	}
}

// getEnv gets an environment variable or returns a default value
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getEnvDuration gets a duration from an environment variable or returns a default
func getEnvDuration(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}
