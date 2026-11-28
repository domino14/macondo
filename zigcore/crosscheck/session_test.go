package crosscheck

import (
	"strings"
	"testing"
)

func startEngine(t *testing.T) *Engine {
	t.Helper()
	path, err := BinaryPath()
	if err != nil {
		t.Fatal(err)
	}
	if path == "" {
		t.Skip("no Zig engine binary; run `zig build` in zigcore or set MACONDO_ZIG_ENGINE")
	}
	e, err := Start(path)
	if err != nil {
		t.Fatal(err)
	}
	// Quit at the end of the test, failing it if the engine leaked memory.
	t.Cleanup(func() {
		if err := e.Quit(); err != nil {
			t.Error(err)
		}
	})
	return e
}

func TestSession(t *testing.T) {
	e := startEngine(t)

	got, err := e.Command("ucgi", "ucgiok")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(got[0], "id name ") {
		t.Errorf("first line %q, want id name", got[0])
	}
	options := map[string]bool{}
	for _, l := range got {
		if f := strings.Fields(l); len(f) >= 3 && f[0] == "option" && f[1] == "name" {
			options[f[2]] = true
		}
	}
	for _, want := range []string{"threads", "hash-mb", "data-path"} {
		if !options[want] {
			t.Errorf("option %s missing from %q", want, got)
		}
	}

	// A valid setoption is silent, so isready's answer is the only line.
	if err := e.Send("setoption name threads value 4"); err != nil {
		t.Fatal(err)
	}
	if got, err := e.Command("isready", "readyok"); err != nil || len(got) != 1 {
		t.Errorf("setoption + isready: %q, %v", got, err)
	}

	got, err = e.Command("frobnicate", "error")
	if err != nil {
		t.Fatal(err)
	}
	if want := "error unknown-command frobnicate"; got[len(got)-1] != want {
		t.Errorf("got %q, want %q", got[len(got)-1], want)
	}

	if err := e.Quit(); err != nil {
		t.Errorf("quit: %v", err)
	}
}
