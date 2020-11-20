package main

import (
	"errors"
	"sync"
	"sync/atomic"
	"syscall/js"
	"unsafe"

	"github.com/domino14/macondo/analyzer"
	"github.com/domino14/macondo/cache"
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

// (int32, string) => string
func analyzerAnalyze(this js.Value, args []js.Value) (interface{}, error) {
	k := int32(args[0].Int())
	thing, loaded := analyzerMap.Load(k)
	if !loaded {
		return nil, errors.New("invalid id")
	}
	an, ok := thing.(*analyzer.Analyzer)
	if !ok {
		return nil, errors.New("invalid type")
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

func registerCallbacks() {
	js.Global().Get("resMacondo").Invoke(map[string]interface{}{
		"precache":        js.FuncOf(precache),
		"newAnalyzer":     js.FuncOf(asyncFunc(newAnalyzer)),
		"delAnalyzer":     js.FuncOf(asyncFunc(delAnalyzer)),
		"analyzerAnalyze": js.FuncOf(asyncFunc(analyzerAnalyze)),
	})
}

func main() {
	registerCallbacks()
	// Keep Go "program" running.
	select {}
}
