package main

import (
	"fmt"
	"syscall/js"
)

// src must be Uint8Array or Uint8ClampedArray
func readJSBytes(src js.Value) []byte {
	srcLen := src.Length()
	dst := make([]byte, srcLen)
	dstLen := js.CopyBytesToGo(dst, src)
	if dstLen != srcLen {
		js.Global().Get("console").Get("warn").Invoke(fmt.Sprintf("copied %d js bytes into %d go bytes", srcLen, dstLen))
	}
	return dst
}

func makeJSBytes(src []byte) js.Value {
	srcLen := len(src)
	dst := js.Global().Get("Uint8Array").New(srcLen)
	dstLen := js.CopyBytesToJS(dst, src)
	if dstLen != srcLen {
		js.Global().Get("console").Get("warn").Invoke(fmt.Sprintf("copied %d go bytes into %d js bytes", srcLen, dstLen))
	}
	return dst
}

func asyncFunc(f func(js.Value, []js.Value) (interface{}, error)) func(js.Value, []js.Value) interface{} {
	return func(this js.Value, args []js.Value) interface{} {
		return js.Global().Get("Promise").New(js.FuncOf(func(_ js.Value, settlers []js.Value) interface{} {
			res, rej := settlers[0], settlers[1]
			ret, err := f(this, args)
			if err != nil {
				// throw a string.
				return rej.Invoke(err.Error())
			} else {
				return res.Invoke(ret)
			}
		}))
	}
}
