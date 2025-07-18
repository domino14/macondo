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

// BestSimmingMove finds the best simming move for a given CGP and plies using Magpie.
// numPlays is the max number of plays to simulate.
func (m *Magpie) BestSimmingMove(cgp string, lex string, plies int, numPlays int) (string, string) {
	threads := runtime.NumCPU()

	// Prepare command options as key-value pairs
	cmdOptions := map[string]interface{}{
		"s1":            "equity",
		"s2":            "equity",
		"r1":            "equity",
		"r2":            "equity",
		"numplays":      numPlays,
		"plies":         plies,
		"maxequitydiff": 30,
		"sr":            "tt", // top-two condition for BAI
		"minp":          100,  // BAI: minimum iterations considered per play
		"threads":       threads,
		"thres":         "gk16",
		"scond":         95,     // BAI stopping condition (percent confidence)
		"it":            100000, // BAI: max TOTAL iterations across all plays.
		"wmp":           "true",
		"lex":           lex,
	}

	// Helper to build command string from map
	buildCmd := func(options map[string]interface{}) string {
		var buf bytes.Buffer
		for k, v := range options {
			buf.WriteString(fmt.Sprintf("-%s %v ", k, v))
		}
		return buf.String()
	}

	settings := buildCmd(cmdOptions)
	CaptureCStdout(func() {
		sendCmd("set " + settings)
	})
	CaptureCStdout(func() {
		sendCmd("cgp " + cgp)
	})
	log.Info().Str("cgp", cgp).Str("settings", settings).
		Msg("Running Magpie gen moves")
	CaptureCStdout(func() {
		sendCmd("gen")
	})
	log.Info().Msg("Running Magpie simming move")

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
	// tmpFile, err := os.CreateTemp("", bestMove+"_*")
	// if err != nil {
	// 	log.Error().Err(err).Msg("Failed to create temp file")
	// } else {
	// 	defer tmpFile.Close()
	// 	_, err = tmpFile.WriteString(output)
	// 	if err != nil {
	// 		log.Error().Err(err).Msg("Failed to write to temp file")
	// 	}
	// }

	return bestMove, output
}

// SanityTest
func (m *Magpie) SanityTest() {
	// cgp := "C14/O2TOY9/mIRADOR8/F4DAB2PUGH1/I5GOOEY3V/T4XI2MALTHA/14N/6GUM3OWN/7PEW2DOE/9EF1DOR/2KUNA1J1BEVELS/3TURRETs2S2/7A4T2/7N7/7S7 EEEIILZ/ 336/298 0 -lex NWL23"
	// bestMoves := map[string]int{}
	// for i := range 100 {
	// 	res := m.BestSimmingMove(cgp, 5, 100)
	// 	bestMoves[res] += 1
	// 	if i%10 == 0 {
	// 		log.Info().Int("iteration", i).Msg("Best simming move iteration")
	// 	}
	// }
	// log.Info().Interface("bestMoves", bestMoves).Msg("Best simming move found")
	cgp := "DO13/EX13/L9P1F2/V9W1O2/EL8N1I2/1A1FOB1OBJETS2/1LEEK3A1D1T2/EL1MACHER6/PA6T6/ON1G1WAGED2Q2/x2O2YIN3U2/I1ZA4D3I2/E1I9R2/SUGARY4HAEN1/5ATrIUMS3 EIIOORT/ 360/331 0 -lex NWL23 -ld english"
	res, _ := m.BestSimmingMove(cgp, "NWL23", 11, 80)
	log.Info().Str("cgp", cgp).Str("bestMove", res).Msg("Best simming move found")
}
