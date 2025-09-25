# PEG Log Viewer

A terminal-based viewer for Macondo PEG (Pre-Endgame) solver logs with tree-based format.

## Features

- **Collapsible tree view** of recursive decision trees
- **Color-coded outcomes** (green=win, red=loss/cutoff, yellow=search matches)
- **Search functionality** to find specific moves or patterns
- **Detailed info panel** showing timing, game state, and statistics
- **Keyboard navigation** for efficient browsing

## Usage

```bash
# Build the viewer
go build

# Run with a log file
./pegviewer /path/to/macondo-peglog

# Or use directly with go run
go run main.go /path/to/macondo-peglog
```

## Navigation

- **↑/↓** or **j/k** - Move up/down in tree
- **→/l** or **Space** - Expand node  
- **←/h** - Collapse node
- **/** - Start search
- **q** - Quit
- **h** - Show help

## Log Format

The viewer expects YAML logs with the new tree-based format:

```yaml
job_id: "job_001"
peg_play: "11D ...D"
meta:
  thread: 0
  timestamp: "2024-02-19T23:23:21-05:00"
  bag_state: "AC"
  empties_bag: false
options:
  - perm_in_bag: "AC"
    perm_ct: 1
    execution_tree:
      root:
        depth: 0
        player: 1
        move: "E5 NOH"
        game_state:
          spread: -5
          outcome_before: "not_initialized"
          outcome_after: "win"
        timing:
          duration_ms: 15
        children:
          - depth: 1
            player: 0
            move: "3L .O.."
            # ... more children
statistics:
  total_nodes: 45
  max_depth: 3
  cutoffs: 12
  endgames_solved: 8
```

## Colors

- **Green** - Winning outcomes
- **Red** - Losing outcomes / Early cutoffs  
- **Yellow** - Search matches
- **Blue** - Metadata sections
- **Purple** - Execution tree sections
- **White** - Default text

## Tips

- Use search (`/`) to quickly find specific moves or outcomes
- Expand interesting branches to see the full decision tree
- The info panel shows detailed timing and state information
- Look for red nodes to identify early cutoffs and optimizations