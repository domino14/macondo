package main

import (
	"errors"
	"sync"
	"sync/atomic"
	"syscall/js"
	"unsafe"

	"github.com/domino14/macondo/analyzer"
	"github.com/domino14/macondo/cache"
	"github.com/rs/zerolog"
)

func precache(this js.Value, args []js.Value) interface{} {
	cache.Precache(args[0].String(), readJSBytes(args[1]))
	return nil
}

var analyzerMap sync.Map
var analyzerLastId int32 // int64 does not fit js's float64.

// () => int32
func newAnalyzer(this js.Value, args []js.Value) (interface{}, error) {
	an := analyzer.NewDefaultAnalyzer()
	k := atomic.AddInt32(&analyzerLastId, 1)
	_, loaded := analyzerMap.LoadOrStore(k, an)
	if loaded {
		return nil, errors.New("atomicity issue")
	}
	return k, nil
}

// (int32) => null
func delAnalyzer(this js.Value, args []js.Value) (interface{}, error) {
	k := int32(args[0].Int())
	_, loaded := analyzerMap.LoadAndDelete(k)
	if !loaded {
		return nil, errors.New("invalid id")
	}
	return nil, nil
}

func getAnalyzer(k int32) (*analyzer.Analyzer, error) {
	thing, loaded := analyzerMap.Load(k)
	if !loaded {
		return nil, errors.New("invalid id")
	}
	an, ok := thing.(*analyzer.Analyzer)
	if !ok {
		return nil, errors.New("invalid type")
	}
	return an, nil
}

// (int32, string) => string
func analyzerAnalyze(this js.Value, args []js.Value) (interface{}, error) {
	an, err := getAnalyzer(int32(args[0].Int()))
	if err != nil {
		return nil, err
	}

	// JS doesn't use utf8, but it converts automatically if we take/return strings.
	jsonBoardStr := args[1].String()
	jsonBoard := *(*[]byte)(unsafe.Pointer(&jsonBoardStr))

	jsonMoves, err := an.Analyze(jsonBoard)
	if err != nil {
		return nil, err
	}
	jsonMovesStr := *(*string)(unsafe.Pointer(&jsonMoves))
	return jsonMovesStr, nil
}

// (int32) => null
func simInit(this js.Value, args []js.Value) (interface{}, error) {
	an, err := getAnalyzer(int32(args[0].Int()))
	if err != nil {
		return nil, err
	}

	err = an.SimInit()
	return nil, err
}

// (int32, int) => null
func simSingleThread(this js.Value, args []js.Value) (interface{}, error) {
	an, err := getAnalyzer(int32(args[0].Int()))
	if err != nil {
		return nil, err
	}

	err = an.SimSingleThread(args[1].Int())
	return nil, err
}

// (int32) => not documented
func simState(this js.Value, args []js.Value) (interface{}, error) {
	an, err := getAnalyzer(int32(args[0].Int()))
	if err != nil {
		return nil, err
	}

	retBytes, err := an.SimState()
	if err != nil {
		return nil, err
	}

	retStr := *(*string)(unsafe.Pointer(&retBytes))
	return retStr, nil
}

func registerCallbacks() {
	js.Global().Get("resMacondo").Invoke(map[string]interface{}{
		"precache":        js.FuncOf(precache),
		"newAnalyzer":     js.FuncOf(asyncFunc(newAnalyzer)),
		"delAnalyzer":     js.FuncOf(asyncFunc(delAnalyzer)),
		"analyzerAnalyze": js.FuncOf(asyncFunc(analyzerAnalyze)),
		"simInit":         js.FuncOf(asyncFunc(simInit)),
		"simSingleThread": js.FuncOf(asyncFunc(simSingleThread)),
		"simState":        js.FuncOf(asyncFunc(simState)),
	})
}

func main() {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	registerCallbacks()
	// Keep Go "program" running.
	select {}
}
