package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/domino14/macondo/shell"
)

func main() {
	if len(os.Args) < 4 {
		fmt.Println("Usage: updater <exe-path> <extracted-dir> <data-dir>")
		os.Exit(1)
	}

	exePath := os.Args[1]
	extractedDir := os.Args[2]
	dataDir := os.Args[3]

	// Wait for the old executable to exit
	time.Sleep(2 * time.Second)

	// Loop until the old executable is no longer running
	for isProcessRunning(exePath) {
		time.Sleep(1 * time.Second)
	}

	// Replace the old executable with the new one
	newExePath := filepath.Join(extractedDir, filepath.Base(exePath))
	err := os.Remove(exePath)
	if err != nil {
		fmt.Println("Failed to remove old executable:", err)
		os.Exit(1)
	}

	err = os.Rename(newExePath, exePath)
	if err != nil {
		fmt.Println("Failed to replace executable:", err)
		os.Exit(1)
	}

	// Merge the data directory
	newDataDir := filepath.Join(extractedDir, "data")
	err = shell.MergeDirectories(newDataDir, dataDir)
	if err != nil {
		fmt.Println("Failed to merge data directories:", err)
		os.Exit(1)
	}

	// Restart the application
	cmd := exec.Command(exePath)
	err = cmd.Start()
	if err != nil {
		fmt.Println("Failed to restart application:", err)
		os.Exit(1)
	}
}

func isProcessRunning(exePath string) bool {
	exeName := filepath.Base(exePath)
	cmd := exec.Command("tasklist", "/FI", fmt.Sprintf("IMAGENAME eq %s", exeName))
	output, err := cmd.Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(output), exeName)
}
