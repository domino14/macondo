// Package crosscheck runs the Zig engine core over UCGI and compares its
// answers with the Go engine. See zigcore/PLAN.md, "Testing".
//
// The tests find the engine binary through MACONDO_ZIG_ENGINE, or at
// zigcore/zig-out/bin/macondo-zig after `zig build`. If MACONDO_ZIG_ENGINE
// is unset and no binary is built, the tests skip, so `go test ./...` works
// without Zig installed.
package crosscheck

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// DefaultTimeout bounds how long Engine waits for an expected response.
const DefaultTimeout = 10 * time.Second

// Engine is a UCGI session with a child engine process.
type Engine struct {
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	lines   chan string
	readErr chan error
	Timeout time.Duration
}

// BinaryPath returns the engine binary to test, or "" if none is available
// and the tests should skip. It errors if MACONDO_ZIG_ENGINE names a missing
// file, so CI can't silently skip.
func BinaryPath() (string, error) {
	if p := os.Getenv("MACONDO_ZIG_ENGINE"); p != "" {
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("MACONDO_ZIG_ENGINE: %w", err)
		}
		return p, nil
	}
	// Tests run in the package directory, zigcore/crosscheck.
	p, err := filepath.Abs(filepath.Join("..", "zig-out", "bin", "macondo-zig"))
	if err != nil {
		return "", err
	}
	if _, err := os.Stat(p); err != nil {
		return "", nil
	}
	return p, nil
}

// Start launches the engine at path. The engine's stderr goes to stderr.
func Start(path string, args ...string) (*Engine, error) {
	cmd := exec.Command(path, args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	e := &Engine{
		cmd:     cmd,
		stdin:   stdin,
		lines:   make(chan string, 1024),
		readErr: make(chan error, 1),
		Timeout: DefaultTimeout,
	}
	go e.readLoop(stdout)
	return e, nil
}

func (e *Engine) readLoop(r io.Reader) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for sc.Scan() {
		e.lines <- sc.Text()
	}
	err := sc.Err()
	if err == nil {
		err = io.EOF
	}
	e.readErr <- err
	close(e.lines)
}

// Send writes one command line to the engine.
func (e *Engine) Send(line string) error {
	_, err := io.WriteString(e.stdin, line+"\n")
	return err
}

// ReadLine returns the engine's next output line.
func (e *Engine) ReadLine() (string, error) {
	select {
	case line, ok := <-e.lines:
		if !ok {
			return "", <-e.readErr
		}
		return line, nil
	case <-time.After(e.Timeout):
		return "", errors.New("timed out waiting for the engine")
	}
}

// ReadUntil reads lines up to and including the first one whose first
// token is final, and returns them all.
func (e *Engine) ReadUntil(final string) ([]string, error) {
	var got []string
	for {
		line, err := e.ReadLine()
		if err != nil {
			return got, fmt.Errorf("waiting for %q: %w", final, err)
		}
		got = append(got, line)
		if first, _, _ := strings.Cut(line, " "); first == final {
			return got, nil
		}
	}
}

// Command sends line and returns the response lines up to and including the
// line starting with final.
func (e *Engine) Command(line, final string) ([]string, error) {
	if err := e.Send(line); err != nil {
		return nil, err
	}
	return e.ReadUntil(final)
}

// Quit sends `quit` and waits for the process to exit, killing it if it
// doesn't within the timeout.
func (e *Engine) Quit() error {
	_ = e.Send("quit")
	_ = e.stdin.Close()
	done := make(chan error, 1)
	go func() { done <- e.cmd.Wait() }()
	select {
	case err := <-done:
		return err
	case <-time.After(e.Timeout):
		_ = e.cmd.Process.Kill()
		<-done
		return errors.New("engine did not exit after quit; killed")
	}
}
