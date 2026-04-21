// Package notebook implements the Macondo Notebook HTTP server.
// It exposes a REST + SSE API consumed by the notebook web frontend.
//
// Protocol:
//
//	POST /api/execute  { cell_id, content, language } -> 202 Accepted
//	POST /api/cancel   { cell_id }                    -> 200 OK
//	GET  /api/stream                                  -> text/event-stream
//	GET  /                                            -> notebook SPA (React app or placeholder)
package notebook

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"strings"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/shell"
)

// ExecuteRequest is the JSON body for POST /api/execute.
type ExecuteRequest struct {
	CellID   string `json:"cell_id"`
	Content  string `json:"content"`
	Language string `json:"language"` // "macondo", "lua", "markdown"
}

// CancelRequest is the JSON body for POST /api/cancel.
type CancelRequest struct {
	CellID string `json:"cell_id"`
}

// SSEEvent is the structure sent over the SSE stream.
type SSEEvent struct {
	CellID string               `json:"cell_id"`
	Output *shell.NotebookOutput `json:"output,omitempty"`
	Done   bool                  `json:"done"`
	Error  string               `json:"error,omitempty"`
}

// frontendFS embeds the compiled React frontend.
// Run `npm run build` in notebook/frontend/ to populate frontend/dist/.
//
//go:embed frontend/dist
var frontendFS embed.FS

// Server is the notebook HTTP server.
type Server struct {
	sc      *shell.ShellController
	clients map[string]chan SSEEvent
	mu      sync.RWMutex

	// cancelFns maps cell_id to a cancel function for in-progress executions.
	cancelFns map[string]func()
	cancelMu  sync.Mutex
}

// NewServer creates a new notebook server wrapping a ShellController.
func NewServer(cfg *config.Config, execPath, gitVersion string) *Server {
	sc := shell.NewShellController(cfg, execPath, gitVersion)
	return &Server{
		sc:        sc,
		clients:   make(map[string]chan SSEEvent),
		cancelFns: make(map[string]func()),
	}
}

// Handler returns the HTTP handler for the notebook server.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/stream", s.handleSSE)
	mux.HandleFunc("/api/execute", s.handleExecute)
	mux.HandleFunc("/api/cancel", s.handleCancel)
	mux.HandleFunc("/api/state", s.handleState)

	// Serve built frontend assets (fonts, JS bundles, etc.)
	sub, err := fs.Sub(frontendFS, "frontend/dist")
	if err == nil {
		mux.Handle("/assets/", http.FileServer(http.FS(sub)))
	}

	// All other paths serve the SPA index (or placeholder if not built yet).
	mux.HandleFunc("/", s.handleIndex)
	return corsMiddleware(mux)
}

// broadcast sends an SSE event to all connected clients.
func (s *Server) broadcast(event SSEEvent) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, ch := range s.clients {
		select {
		case ch <- event:
		default: // drop if buffer full
		}
	}
}

// handleSSE handles GET /api/stream — the SSE endpoint.
func (s *Server) handleSSE(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	clientID := fmt.Sprintf("%p", r)
	ch := make(chan SSEEvent, 64)
	s.mu.Lock()
	s.clients[clientID] = ch
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.clients, clientID)
		s.mu.Unlock()
	}()

	// Send a keep-alive comment on connect.
	fmt.Fprintf(w, ": connected\n\n")
	flusher.Flush()

	for {
		select {
		case event := <-ch:
			data, err := json.Marshal(event)
			if err != nil {
				continue
			}
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		case <-r.Context().Done():
			return
		}
	}
}

// handleExecute handles POST /api/execute.
// Execution runs asynchronously; outputs flow over SSE.
func (s *Server) handleExecute(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ExecuteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.CellID == "" {
		http.Error(w, "cell_id is required", http.StatusBadRequest)
		return
	}

	// Markdown cells produce no server-side output.
	if req.Language == "markdown" {
		s.broadcast(SSEEvent{CellID: req.CellID, Done: true})
		w.WriteHeader(http.StatusAccepted)
		return
	}

	go s.executeCell(req)
	w.WriteHeader(http.StatusAccepted)
}

// executeCell runs the cell content line by line, broadcasting outputs over SSE.
// For long-running operations (sim, endgame, peg) it streams progress events.
// The cell's context is stored so POST /api/cancel can stop execution mid-flight.
func (s *Server) executeCell(req ExecuteRequest) {
	ctx, cancel := context.WithCancel(context.Background())

	s.cancelMu.Lock()
	s.cancelFns[req.CellID] = cancel
	s.cancelMu.Unlock()

	defer func() {
		cancel()
		s.cancelMu.Lock()
		delete(s.cancelFns, req.CellID)
		s.cancelMu.Unlock()
	}()

	// outCh collects outputs; a dedicated goroutine broadcasts them.
	outCh := make(chan *shell.NotebookOutput, 64)
	broadcastDone := make(chan struct{})
	go func() {
		defer close(broadcastDone)
		for out := range outCh {
			s.broadcast(SSEEvent{CellID: req.CellID, Output: out})
		}
	}()

	var execErr error
	lines := strings.Split(req.Content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		if err := s.sc.NotebookExecuteStreaming(ctx, line, outCh); err != nil {
			if !errors.Is(err, context.Canceled) {
				log.Error().Str("cell_id", req.CellID).Str("line", line).Err(err).Msg("notebook execute error")
				outCh <- &shell.NotebookOutput{Kind: "error", Data: err.Error()}
			}
			execErr = err
			break
		}
	}

	close(outCh)
	<-broadcastDone // ensure all outputs are flushed before Done

	if execErr == nil || errors.Is(execErr, context.Canceled) {
		s.broadcast(SSEEvent{CellID: req.CellID, Done: true})
	} else {
		s.broadcast(SSEEvent{CellID: req.CellID, Done: true, Error: execErr.Error()})
	}
}

// handleCancel handles POST /api/cancel.
func (s *Server) handleCancel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req CancelRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.cancelMu.Lock()
	if cancel, ok := s.cancelFns[req.CellID]; ok {
		cancel()
		delete(s.cancelFns, req.CellID)
	}
	s.cancelMu.Unlock()
	w.WriteHeader(http.StatusOK)
}

// StateResponse is returned by GET /api/state.
type StateResponse struct {
	GameLoaded bool `json:"game_loaded"`
}

// handleState handles GET /api/state — lightweight poll for game status.
func (s *Server) handleState(w http.ResponseWriter, r *http.Request) {
	resp := StateResponse{
		GameLoaded: s.sc.GameLoaded(),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleIndex serves the notebook SPA.
// After running `npm run build` in notebook/frontend/, the compiled React app
// is embedded and served here. Before that, the placeholder HTML is shown.
func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	content, err := frontendFS.ReadFile("frontend/dist/index.html")
	if err != nil {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, placeholderHTML)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(content)
}

// corsMiddleware adds permissive CORS headers for local development.
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

const placeholderHTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Macondo Notebook</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #1a1a2e; color: #e0e0e0; min-height: 100vh; }
    header { background: #16213e; padding: 16px 24px; border-bottom: 1px solid #0f3460;
             display: flex; align-items: center; gap: 12px; }
    header h1 { font-size: 1.3rem; font-weight: 600; color: #e94560; letter-spacing: 1px; }
    header span { font-size: 0.8rem; color: #888; }
    .container { max-width: 900px; margin: 40px auto; padding: 0 24px; }
    .notice { background: #16213e; border: 1px solid #0f3460; border-radius: 8px;
              padding: 24px; margin-bottom: 24px; }
    .notice h2 { color: #e94560; margin-bottom: 12px; }
    .notice p { color: #aaa; line-height: 1.6; margin-bottom: 8px; }
    .notice code { background: #0f3460; padding: 2px 6px; border-radius: 4px;
                   font-family: "SF Mono", Consolas, monospace; font-size: 0.9em; color: #64dfdf; }
    .cell { background: #16213e; border: 1px solid #0f3460; border-radius: 8px;
            margin-bottom: 16px; overflow: hidden; }
    .cell-header { background: #0f3460; padding: 8px 16px; font-size: 0.8rem; color: #888;
                   display: flex; justify-content: space-between; align-items: center; }
    .cell-input { padding: 16px; font-family: "SF Mono", Consolas, monospace;
                  font-size: 0.9rem; color: #64dfdf; white-space: pre; }
    .cell-output { border-top: 1px solid #0f3460; padding: 16px;
                   font-family: "SF Mono", Consolas, monospace; font-size: 0.85rem; color: #aaa; }
    .run-btn { background: #e94560; color: white; border: none; border-radius: 4px;
               padding: 6px 14px; cursor: pointer; font-size: 0.8rem; }
    .run-btn:hover { background: #c73652; }
    #status { font-size: 0.8rem; color: #888; margin-top: 8px; }
    #output-area { margin-top: 8px; }
    .event-row { padding: 4px 0; border-bottom: 1px solid #0f3460; }
    .event-kind { color: #e94560; font-size: 0.75rem; text-transform: uppercase; margin-right: 8px; }
  </style>
</head>
<body>
  <header>
    <h1>⚡ Macondo Notebook</h1>
    <span>Research interface — frontend coming in Phase 2</span>
  </header>

  <div class="container">
    <div class="notice">
      <h2>Phase 1 API is live</h2>
      <p>The Go backend is running. Connect to the SSE stream at
        <code>GET /api/stream</code>, then POST commands to
        <code>POST /api/execute</code> with JSON body
        <code>{"cell_id": "1", "content": "load ...\ngen 15"}</code>.
      </p>
      <p>The React frontend (Phase 2) will replace this page with a full notebook UI
         including an interactive 2D board, move tables, and charts.</p>
    </div>

    <div class="cell">
      <div class="cell-header">
        <span>Try it — Macondo command cell</span>
        <button class="run-btn" onclick="runDemo()">▶ Run</button>
      </div>
      <div class="cell-input" id="demo-input">new
gen 15</div>
      <div class="cell-output" id="demo-output">(output will appear here)</div>
    </div>

    <div id="status"></div>
  </div>

  <script>
    const cellID = 'demo-' + Date.now();
    const outputEl = document.getElementById('demo-output');
    const statusEl = document.getElementById('status');

    // Connect to SSE stream
    const es = new EventSource('/api/stream');
    es.onmessage = (e) => {
      const event = JSON.parse(e.data);
      if (event.cell_id !== cellID) return;
      if (event.error) {
        outputEl.innerHTML += '<div style="color:#e94560">Error: ' + escHtml(event.error) + '</div>';
      } else if (event.output) {
        const kind = event.output.kind;
        const data = event.output.data;
        if (kind === 'text') {
          outputEl.innerHTML += '<pre style="white-space:pre-wrap">' + escHtml(data) + '</pre>';
        } else if (kind === 'board') {
          outputEl.innerHTML += '<div><span class="event-kind">board</span>FEN: ' +
            escHtml(data.fen) + ' | Turn: ' + data.turnNumber + '</div>';
        } else if (kind === 'table') {
          let html = '<table style="border-collapse:collapse;width:100%"><thead><tr>' +
            '<th style="text-align:left;padding:4px 8px;color:#e94560">Rank</th>' +
            '<th style="text-align:left;padding:4px 8px;color:#e94560">Move</th>' +
            '<th style="text-align:right;padding:4px 8px;color:#e94560">Score</th>' +
            '<th style="text-align:right;padding:4px 8px;color:#e94560">Equity</th>' +
            '</tr></thead><tbody>';
          for (const row of (data.moves || [])) {
            html += '<tr><td style="padding:2px 8px">' + row.rank + '</td>' +
              '<td style="padding:2px 8px;font-family:monospace">' + escHtml(row.move) + '</td>' +
              '<td style="padding:2px 8px;text-align:right">' + row.score + '</td>' +
              '<td style="padding:2px 8px;text-align:right">' + row.equity.toFixed(2) + '</td></tr>';
          }
          html += '</tbody></table>';
          outputEl.innerHTML += html;
        } else {
          outputEl.innerHTML += '<div><span class="event-kind">' + escHtml(kind) + '</span>' +
            '<pre style="display:inline;white-space:pre-wrap">' +
            escHtml(JSON.stringify(data, null, 2)) + '</pre></div>';
        }
      }
      if (event.done) {
        statusEl.textContent = 'Done.';
      }
    };
    es.onerror = () => { statusEl.textContent = 'SSE connection error.'; };

    async function runDemo() {
      const content = document.getElementById('demo-input').textContent.trim();
      outputEl.innerHTML = '';
      statusEl.textContent = 'Running...';
      await fetch('/api/execute', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cell_id: cellID, content, language: 'macondo'}),
      });
    }

    function escHtml(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
  </script>
</body>
</html>
`
