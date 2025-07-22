# Macondo TUI

A modern terminal user interface for Macondo crossword game analysis.

## Features

### Core Interface
- **Two-panel layout**: Board display on the left, analysis tools on the right
- **Board visualization**: Full crossword board with premium squares marked
- **Player rack display**: Current player's tiles with scores and turn information
- **Bag analysis**: Unseen tiles breakdown with vowel/consonant statistics

### Navigation & Controls
- **Keyboard shortcuts**:
  - `n` - Next turn
  - `p` - Previous turn
  - `g` - Generate moves (15 top moves)
  - `l` - Load game dialog
  - `s` - Start simulation (coming soon)
  - `e` - Endgame solver (coming soon)
  - `q` - Quit

- **Button interface**: Click buttons for the same functionality

### Game Loading Options
- **GCG files**: Load standard crossword game files
- **CGP positions**: Load specific board positions
- **Cross-tables games**: Load games by ID from cross-tables.com
- **Woogles games**: Load games by ID from woogles.io
- **New games**: Create fresh games with random first player

### Move Analysis
- **Move generation**: Generate and rank top 15 moves by equity
- **Move display**: Shows move notation, leave, score, and equity
- **Real-time updates**: All panels refresh when navigating through game

## Usage

### Building
```bash
make macondo_tui
```

### Running
```bash
./bin/tui
```

### Quick Start
1. Launch the TUI: `./bin/tui`
2. Press 'l' to open the load dialog
3. Choose "New Game" to start a fresh game
4. Press 'g' to generate moves for the current position
5. Use 'n' and 'p' to navigate through the game
6. Press 'q' to quit

### Loading Games
- **File**: Select this to load a local GCG file
- **CGP**: Enter a CGP position string for analysis
- **Cross-tables**: Enter a game ID from cross-tables.com (e.g., "19009")
- **Woogles**: Enter a game ID from woogles.io
- **New Game**: Start with random players and a fresh board

## Architecture

The TUI is built using:
- **tview**: Modern terminal UI framework with rich widgets
- **Shell integration**: Reuses all the existing Macondo shell functionality
- **Modular panels**: Separate components for board, rack, bag, and controls
- **Real-time updates**: Automatic refresh of all panels on state changes

## Future Enhancements

Planned features:
- Monte Carlo simulation with progress display
- Endgame solver integration
- Pre-endgame (PEG) analysis
- Move entry and gameplay
- Game export functionality
- Settings and preferences panel
- Help system integration

## Development

The TUI code is organized in `/tui/`:
- `app.go` - Main application controller
- `board.go` - Board display panel
- `rack.go` - Player rack and scores
- `bag.go` - Tile bag analysis
- `controls.go` - Buttons and move list
- `dialogs.go` - Load game dialogs

Shell integration is provided through `shell/tui_methods.go` which exposes clean methods for TUI interaction with the existing shell controller.