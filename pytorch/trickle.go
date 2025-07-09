// trickle.go – reads 1 KiB per second
package main

import (
	"io"
	"os"
	"time"

	"github.com/rs/zerolog/log"
)

func main() {
	buf := make([]byte, 1024) // 1 KiB buffer
	log.Info().Msg("Starting trickle consumer...")
	for {
		_, err := os.Stdin.Read(buf)
		if err == io.EOF {
			log.Info().Msg("End of input stream, exiting...")
			return // upstream closed
		}
		if err != nil {
			log.Error().Err(err).Msg("Error reading from stdin")
			panic(err)
		}
		time.Sleep(1 * time.Second) // throttle
	}
}
