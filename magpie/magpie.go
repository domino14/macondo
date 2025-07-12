package magpie

/*
#cgo CFLAGS: -I../../magpie/src
#cgo LDFLAGS: -L../../magpie -lmagpie -lm
#include "impl/wasm_api.h"
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

// Helper to flush C stdio buffers
void flush_c_stdout() { fflush(stdout); }
*/
import "C"
import (
	"bytes"
	"fmt"
	"os"
	"runtime"
	"syscall"
	"unsafe"

	"github.com/domino14/macondo/config"
	"github.com/rs/zerolog/log"
)

type Magpie struct {
	configPaths string
}

func NewMagpie(cfg *config.Config) *Magpie {
	cPaths := C.CString(cfg.GetString(config.ConfigMagpieDataPath))
	defer C.free(unsafe.Pointer(cPaths))
	C.wasm_init_configs(cPaths)
	return &Magpie{
		configPaths: cfg.GetString(config.ConfigMagpieDataPath),
	}
}

func sendCmd(cmd string) {
	cCmd := C.CString(cmd)
	defer C.free(unsafe.Pointer(cCmd))
	C.process_command_wasm_sync(cCmd)

}

// CaptureCStdout runs fn() and returns everything written to C stdout as a string.
func CaptureCStdout(fn func()) string {
	// Create a pipe
	r, w, err := os.Pipe()
	if err != nil {
		panic(err)
	}

	// Save original stdout fd
	origStdout, err := syscall.Dup(1)
	if err != nil {
		panic(err)
	}

	// Redirect stdout to pipe
	syscall.Dup2(int(w.Fd()), 1)

	// Run the function (which calls C code)
	fn()

	// Flush C stdio buffers
	C.flush_c_stdout()

	// Restore original stdout
	w.Close()
	syscall.Dup2(origStdout, 1)
	syscall.Close(origStdout)

	// Read output
	var buf bytes.Buffer
	_, _ = buf.ReadFrom(r)
	r.Close()

	return buf.String()
}

func (m *Magpie) BestSimmingMove(cgp string, plies int) string {
	threads := runtime.NumCPU()

	sendCmd("set -s1 equity -s2 equity -r1 equity -r2 equity -numplays 100 -plies " +
		fmt.Sprint(plies) + " -maxequitydiff 30 -sr tt -threads " +
		fmt.Sprint(threads) + " -thres gk16 -scond 99 -it 1000000 -wmp true")
	sendCmd("cgp " + cgp)
	CaptureCStdout(func() {
		sendCmd("gen")
	})
	output := CaptureCStdout(func() {
		sendCmd("sim")
	})
	// find the string that matches the pattern "bestmove 15f.ELSE or bestmove ex.ABCD or bestmove pass"
	var bestMove string
	outputLines := bytes.Split([]byte(output), []byte("\n"))
	for i := len(outputLines) - 1; i >= 0; i-- {
		line := outputLines[i]
		if bytes.HasPrefix(line, []byte("bestmove ")) {
			move := bytes.TrimPrefix(line, []byte("bestmove "))
			bestMove = string(move)
			break
		}
	}
	return bestMove
}

// SanityTest
func (m *Magpie) SanityTest() {
	cgp := "C14/O2TOY9/mIRADOR8/F4DAB2PUGH1/I5GOOEY3V/T4XI2MALTHA/14N/6GUM3OWN/7PEW2DOE/9EF1DOR/2KUNA1J1BEVELS/3TURRETs2S2/7A4T2/7N7/7S7 EEEIILZ/ 336/298 0 -lex NWL23"
	bestMoves := map[string]int{}
	for range 100 {
		res := m.BestSimmingMove(cgp, 5)
		bestMoves[res] += 1
	}
	log.Info().Interface("bestMoves", bestMoves).Msg("Best simming move found")
}
