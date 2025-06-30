package magpie

/*
#cgo CFLAGS: -I../../magpie/src
#cgo LDFLAGS: -L../../magpie -lmagpie -lm
#include "impl/wasm_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"time"
	"unsafe"
)

func sendCmd(cmd string) {
	cCmd := C.CString(cmd)
	defer C.free(unsafe.Pointer(cCmd))
	C.process_command_wasm(cCmd)

}

func sendCmdWithPoll(cmd string) {
	sendCmd(cmd)
	time.Sleep(500 * time.Millisecond) // Give some time for the command to be processed

	resp := C.get_search_status_wasm()
	fmt.Println("Response from C:", C.GoString(resp))

	defer C.free(unsafe.Pointer(resp))
}

// SanityTest
func SanityTest() {
	paths := "../MAGPIE-DATA/data"
	cPaths := C.CString(paths)
	defer C.free(unsafe.Pointer(cPaths))
	C.wasm_init_configs(cPaths)

	sendCmd("set -s1 equity -s2 equity -r1 equity -r2 equity -numplays 100 -plies 5 -maxequitydiff 30 -sr tt -threads 16 -thres gk16 -scond 99 -it 1000000 -wmp true")
	time.Sleep(500 * time.Millisecond) // Give some time for the command to be processed
	sendCmd("cgp C14/O2TOY9/mIRADOR8/F4DAB2PUGH1/I5GOOEY3V/T4XI2MALTHA/14N/6GUM3OWN/7PEW2DOE/9EF1DOR/2KUNA1J1BEVELS/3TURRETs2S2/7A4T2/7N7/7S7 EEEIILZ/ 336/298 0 -lex NWL23")
	time.Sleep(500 * time.Millisecond) // Give some time for the command to be processed

	sendCmd("gen")
	time.Sleep(500 * time.Millisecond) // Give some time for the command to be processed
	fmt.Println("starting simulation")
	sendCmd("sim")
}
