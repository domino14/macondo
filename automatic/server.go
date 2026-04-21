package automatic

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/rs/zerolog/log"
)

func analyzeHandler(w http.ResponseWriter, r *http.Request) {
	filename := r.URL.Query().Get("file")
	if filename == "" {
		http.Error(w, "missing 'file' query parameter", http.StatusBadRequest)
		return
	}

	result, err := AnalyzeLogFileData(filename)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	format := r.URL.Query().Get("format")
	if format == "json" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
		return
	}

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	fmt.Fprint(w, FormatTable(result))
}

// StartServer starts an HTTP server on the given address that exposes the
// autoanalyze endpoint. Call as:
//
//	curl 'http://host:8080/analyze?file=games.txt'
//	curl 'http://host:8080/analyze?file=games.txt&format=json'
func StartServer(addr string) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/analyze", analyzeHandler)
	log.Info().Str("addr", addr).Msg("starting analyze server")
	return http.ListenAndServe(addr, mux)
}
