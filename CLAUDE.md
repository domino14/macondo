# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Macondo is a high-performance crossword board game solver (Scrabble™, Words with Friends, etc.) written in Go. It features advanced AI capabilities including Monte Carlo tree search, endgame solving, and neural network integration.

## Common Development Commands

### Building
```bash
# Build the interactive shell
make macondo_shell

# Build the bot
make macondo_bot

# Build all binaries
make all

# Build for AWS Lambda
make lambda
```

### Testing
```bash
# Run all tests
go test ./...

# Run tests for a specific package
go test ./game/...

# Run tests with verbose output
go test -v ./...

# Run a specific test
go test -v -run TestSpecificFunction ./package/...

# Run benchmarks
go test -bench=. ./...
```

### Code Generation
```bash
# Generate protocol buffers
make generate
```

## High-Level Architecture

### Core Components

**Game Engine** (`/game`)
- Central game state management and rules enforcement
- Handles board state, player turns, scoring, and move validation
- Key types: `Game`, `Board`, `Turn`, `Player`

**Move Generation** (`/movegen`)
- Generates all legal moves for a given position
- Integrates with external Wolges solver for high-performance move generation
- Supports both native Go implementation and FFI calls to Rust-based Wolges

**AI System**
- **Bot** (`/ai/bot`): Configurable bot players with different skill levels
- **Monte Carlo** (`/montecarlo`): Tree search for strategic decision making
- **Endgame Solver** (`/endgame/negamax`): Perfect play in endgame positions using negamax with transposition tables
- **Neural Networks**: PyTorch-based CNNs trained on game positions, exported to ONNX for inference

**Equity Calculation** (`/equity`)
- Evaluates position strength considering:
  - Leave values (tiles remaining after a play)
  - Opening and pre-endgame adjustments
  - Neural network evaluations when available

### Data Flow
1. **Input**: Game position (CGP format) or game history (GCG format)
2. **Move Generation**: All legal moves generated via movegen
3. **Evaluation**: Each move evaluated using equity calculators
4. **Selection**: Best move selected based on configured strategy
5. **Execution**: Move applied to game state

### Key Interfaces
- `MoveGenerator`: Interface for different move generation implementations
- `EquityCalculator`: Plugin architecture for position evaluation
- `Bot`: Interface for AI players with varying strategies

### File Formats
- **KWG**: Compressed lexicon format (Kurnia Word Graph)
- **KLV2**: Leave value strategy files
- **GCG**: Game notation for import/export
- **CGP**: Position notation for analysis

### Integration Points
- **Wolges**: External Rust-based move generator via CGO
- **Triton Server**: GPU inference for neural networks in production
- **NATS/WebSocket**: Real-time game communication for bots

## Important Patterns

### Testing
- Table-driven tests are preferred
- Test data lives in `testdata/` directories
- Use `testify/assert` or `matryer/is` for assertions

### Error Handling
- Errors are propagated up the call stack
- Use descriptive error messages with context

### Performance
- Critical paths use efficient data structures (zobrist hashing, bitboards)
- Benchmarks exist for performance-critical code
- GPU acceleration available for neural network inference

### Configuration
- Strategy files and lexicons loaded from `/data` directory
- Bot configurations in JSON format
- Neural network models as ONNX files