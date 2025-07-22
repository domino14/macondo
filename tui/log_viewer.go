package tui

import (
	"io"
	"os"
	"strings"
	"sync"

	"github.com/rivo/tview"
	"github.com/gdamore/tcell/v2"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// LogCapture implements io.Writer to capture log messages
type LogCapture struct {
	messages []string
	mutex    sync.Mutex
	maxLines int
}

func NewLogCapture(maxLines int) *LogCapture {
	return &LogCapture{
		messages: make([]string, 0),
		maxLines: maxLines,
	}
}

func (lc *LogCapture) Write(p []byte) (n int, err error) {
	lc.mutex.Lock()
	defer lc.mutex.Unlock()
	
	message := string(p)
	lc.messages = append(lc.messages, message)
	
	// Keep only the last N messages
	if len(lc.messages) > lc.maxLines {
		lc.messages = lc.messages[len(lc.messages)-lc.maxLines:]
	}
	
	return len(p), nil
}

func (lc *LogCapture) GetMessages() string {
	lc.mutex.Lock()
	defer lc.mutex.Unlock()
	
	return strings.Join(lc.messages, "")
}

func (lc *LogCapture) Clear() {
	lc.mutex.Lock()
	defer lc.mutex.Unlock()
	
	lc.messages = lc.messages[:0]
}

// Add log viewer to TUIApp
func (t *TUIApp) showLogViewer() {
	logText := tview.NewTextView().
		SetDynamicColors(true).
		SetRegions(true).
		SetWrap(true).
		SetScrollable(true)
	
	logText.SetBorder(true).SetTitle("Debug Log (Press ESC to close)")
	
	// Get current log messages
	if t.logCapture != nil {
		logText.SetText(t.logCapture.GetMessages())
	} else {
		logText.SetText("No log capture active")
	}
	
	// Create a flex container to center the log viewer
	flex := tview.NewFlex().
		AddItem(nil, 0, 1, false).
		AddItem(tview.NewFlex().SetDirection(tview.FlexRow).
			AddItem(nil, 0, 1, false).
			AddItem(logText, 0, 3, true).
			AddItem(nil, 0, 1, false), 0, 3, true).
		AddItem(nil, 0, 1, false)
	
	// Set up key handling for the log viewer
	logText.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		if event.Key() == tcell.KeyEscape {
			t.app.SetRoot(t.layout, true).SetFocus(t.controls.GetView())
			return nil
		}
		return event
	})
	
	t.app.SetRoot(flex, true).SetFocus(logText)
}

// Initialize logging with capture
func (t *TUIApp) initLogging() {
	// Create log capture
	t.logCapture = NewLogCapture(1000) // Keep last 1000 lines
	
	// Create temporary log file
	logFile, err := os.CreateTemp("", "macondo-tui-*.log")
	if err != nil {
		// If we can't create the log file, just use the capture
		multiWriter := io.MultiWriter(t.logCapture)
		logger := zerolog.New(multiWriter).With().Timestamp().Logger()
		zerolog.DefaultContextLogger = &logger
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
		return
	}
	
	// Create a multi-writer to send logs to both file and our capture
	multiWriter := io.MultiWriter(logFile, t.logCapture)
	
	// Set up zerolog to write to both and enable debug level
	zerolog.SetGlobalLevel(zerolog.DebugLevel) // Set this FIRST
	logger := zerolog.New(multiWriter).With().Timestamp().Logger()
	
	// Set BOTH the default context logger AND create a global logger
	zerolog.DefaultContextLogger = &logger
	log.Logger = logger
	
	// Test that logging is working with multiple methods
	logger.Info().Str("logfile", logFile.Name()).Msg("TUI logging initialized")
	logger.Debug().Msg("Debug logging is enabled via logger")
	log.Debug().Msg("Debug logging is enabled via log.Debug")
	log.Info().Msg("Testing log.Info as well")
	logFile.Sync() // Force flush to file
}