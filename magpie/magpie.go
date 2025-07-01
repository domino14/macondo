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
	"syscall"
	"unsafe"
)

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

// SanityTest
func SanityTest() {
	paths := "../MAGPIE-DATA/data"
	cPaths := C.CString(paths)
	defer C.free(unsafe.Pointer(cPaths))
	C.wasm_init_configs(cPaths)

	sendCmd("set -s1 equity -s2 equity -r1 equity -r2 equity -numplays 100 -plies 5 -maxequitydiff 30 -sr tt -threads 16 -thres gk16 -scond 99 -it 1000000 -wmp true")
	sendCmd("cgp C14/O2TOY9/mIRADOR8/F4DAB2PUGH1/I5GOOEY3V/T4XI2MALTHA/14N/6GUM3OWN/7PEW2DOE/9EF1DOR/2KUNA1J1BEVELS/3TURRETs2S2/7A4T2/7N7/7S7 EEEIILZ/ 336/298 0 -lex NWL23")
	CaptureCStdout(func() {
		sendCmd("gen")
	})
	output := CaptureCStdout(func() {
		sendCmd("sim")
	})
	fmt.Println("MAGPIE Simulation Output:", output)
}
