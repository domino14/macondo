# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Macondo is a crossword board game solver written in Go, designed to be "the best one in the world." It supports various crossword-style games like Scrabble, Words with Friends, and similar tile-based word games.

## Build and Development Commands

### Core Build Commands
- `make` - Builds all main binaries (shell, bot, bot_shell, analyze, mlproducer)
- `make clean` - Remove all binaries from bin/ directory
- `go generate` - Generate protobuf files (requires protoc-gen-go)

### Individual Binary Builds
- `make macondo_shell` - Interactive shell interface (bin/shell)
- `make macondo_bot` - Bot client (bin/bot)
- `make bot_shell` - Bot shell interface (bin/bot_shell)
- `make analyze` - Analysis tool (bin/analyze)
- `make mlproducer` - Machine learning data producer (bin/mlproducer)

### Testing
- `go test $(go list ./... | grep -v wasm)` - Run all tests except WASM
- `go test ./path/to/package` - Run tests for specific package

### Protobuf Generation
Requires: `go install google.golang.org/protobuf/cmd/protoc-gen-go@latest`
- `protoc --go_out=gen --go_opt=paths=source_relative ./api/proto/macondo/macondo.proto`

## Architecture Overview

### Core Packages

**Game Engine (`game/`)**
- `Game` struct manages complete game state and business logic
- Handles drawing tiles, making moves, scoring, turn management
- Integrates with board, lexicon, cross-sets, and move generation
- Game-agnostic - AI and human players operate at higher levels

**Move Generation (`movegen/`)**
- Generates all legal moves for a given board position and rack
- Uses GADDAG (Directed Acyclic Word Graph) for efficient word finding
- Integrates with cross-set generation for constraint checking

**Board Management (`board/`)**
- `GameBoard` represents the playing surface with premium squares
- Handles tile placement, scoring calculation, and board state
- Cross-set generation for legal tile placement constraints

**AI Components (`ai/`)**
- `ai/bot/` - Bot filters and decision-making logic
- `ai/turnplayer/` - Turn-based player implementations
- `ai/simplesimmer/` - Simple simulation engine

**Lexicon & Word Validation (`lexicon/`)**
- Manages word dictionaries (KWG format from Andy Kurnia's Wolges)
- Handles different lexicons (TWL, CSW, etc.)

**Equity Calculation (`equity/`)**
- Multiple equity calculators for different game phases
- Leave value calculations, opening/endgame/preendgame evaluators
- Supports neural network models via ONNX and Triton

**Move Representation (`move/`, `tinymove/`)**
- `move/` - Full move representation with metadata
- `tinymove/` - Compressed move format for memory efficiency

### Neural Network Integration

The system supports neural network inference through:
- **ONNX Go** (default): `github.com/owulveryck/onnx-go`
- **Triton Server** (recommended): NVIDIA's inference server for better performance
  - Set `MACONDO_TRITON_USE_TRITON=true` environment variable
  - Models stored in `data/strategy/default/models/`

### Command Line Interfaces

**Shell (`cmd/shell/`, `shell/`)**
- Interactive command-line interface for game analysis
- Supports scripting with embedded help system
- Configuration via Viper (YAML/JSON config files)

**Bot (`cmd/bot/`, `bot/`)**
- Automated gameplay client
- Connects to external game servers
- Uses Wolges interface for communication

### Configuration System

Uses Viper for configuration management:
- Config files, environment variables, command-line flags
- Key config: lexicon paths, data directories, AI settings
- Supports relative paths adjusted from executable location

### Data Formats

- **KWG/KLV**: Andy Kurnia's efficient word graph and leave value formats
- **GCG**: Game notation format for importing/exporting games
- **Protobuf**: Internal APIs and data serialization

## Development Notes

- Go 1.24+ required (see go.mod)
- Uses structured logging with zerolog
- Extensive test coverage with table-driven tests
- Memory profiling and CPU profiling support built-in
- CI/CD via GitHub Actions with automated testing and Docker builds