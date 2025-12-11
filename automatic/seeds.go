package automatic

import (
	"bufio"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"os"
	"strings"
)

// GenerateSeeds creates n random 32-byte seeds for deterministic game runs
func GenerateSeeds(n int) ([][32]byte, error) {
	seeds := make([][32]byte, n)
	for i := 0; i < n; i++ {
		_, err := rand.Read(seeds[i][:])
		if err != nil {
			return nil, fmt.Errorf("failed to generate seed %d: %w", i, err)
		}
	}
	return seeds, nil
}

// SaveSeeds writes seeds to a file in base64 format (one per line)
func SaveSeeds(seeds [][32]byte, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create seed file: %w", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// Write header comment
	_, err = writer.WriteString("# Deterministic game seeds (base64 URL-safe encoded, 32 bytes each)\n")
	if err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}

	// Write each seed using URL-safe encoding (avoids / and + characters)
	for i, seed := range seeds {
		encoded := base64.RawURLEncoding.EncodeToString(seed[:])
		_, err = writer.WriteString(encoded + "\n")
		if err != nil {
			return fmt.Errorf("failed to write seed %d: %w", i, err)
		}
	}

	return nil
}

// LoadSeeds reads seeds from a file in base64 format
func LoadSeeds(path string) ([][32]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open seed file: %w", err)
	}
	defer file.Close()

	var seeds [][32]byte
	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Decode base64 seed - try URL-safe encoding first, then standard for backwards compatibility
		decoded, err := base64.RawURLEncoding.DecodeString(line)
		if err != nil {
			// Try standard encoding for old seed files
			decoded, err = base64.RawStdEncoding.DecodeString(line)
			if err != nil {
				return nil, fmt.Errorf("failed to decode seed at line %d: %w", lineNum, err)
			}
		}

		if len(decoded) != 32 {
			return nil, fmt.Errorf("invalid seed length at line %d: got %d bytes, expected 32", lineNum, len(decoded))
		}

		var seed [32]byte
		copy(seed[:], decoded)
		seeds = append(seeds, seed)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading seed file: %w", err)
	}

	return seeds, nil
}
