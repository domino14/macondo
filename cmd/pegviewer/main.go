package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/gdamore/tcell/v2"
	"github.com/rivo/tview"
	"gopkg.in/yaml.v3"
)

// LogData represents the parsed log structure
type LogData struct {
	JobID                string         `yaml:"job_id"`
	PEGPlay              string         `yaml:"peg_play"`
	Meta                 JobMeta        `yaml:"meta"`
	Options              []JobOption    `yaml:"options"`
	Statistics           JobStatistics  `yaml:"statistics"`
	FoundLosses          int            `yaml:"found_losses"`
	MinPotentialLosses   int            `yaml:"min_potential_losses"`
	CutoffAtStart        bool           `yaml:"cutoff_at_start"`
	CutoffWhileIterating bool           `yaml:"cutoff_while_iterating"`
	PEGPlayEmptiesBag    bool           `yaml:"peg_play_empties_bag"`
	EndgamePlies         int            `yaml:"endgame_plies"`
}

type JobMeta struct {
	Thread     int    `yaml:"thread"`
	Timestamp  string `yaml:"timestamp"`
	BagState   string `yaml:"bag_state"`
	EmptiesBag bool   `yaml:"empties_bag"`
}

type JobOption struct {
	PermutationInBag string        `yaml:"perm_in_bag"`
	PermutationCount int           `yaml:"perm_ct"`
	OppRack          string        `yaml:"opp_rack"`
	OurRack          string        `yaml:"our_rack"`
	ExecutionTree    ExecutionTree `yaml:"execution_tree"`
	FinalSpread      int           `yaml:"final_spread"`
	TimeToSolveMs    int64         `yaml:"time_to_solve_ms"`
}

type ExecutionTree struct {
	Root TreeNode `yaml:"root"`
}

type TreeNode struct {
	Depth     int         `yaml:"depth"`
	Player    int         `yaml:"player"`
	Move      string      `yaml:"move"`
	GameState GameState   `yaml:"game_state"`
	Timing    TimingInfo  `yaml:"timing"`
	Cutoff    CutoffInfo  `yaml:"cutoff"`
	Children  []TreeNode  `yaml:"children"`
}

type GameState struct {
	Spread        int    `yaml:"spread"`
	OutcomeBefore string `yaml:"outcome_before"`
	OutcomeAfter  string `yaml:"outcome_after"`
	BagRemaining  int    `yaml:"bag_remaining"`
	GameOver      bool   `yaml:"game_over"`
}

type TimingInfo struct {
	DurationMs int64 `yaml:"duration_ms"`
	StartTime  int64 `yaml:"start_time"`
	EndTime    int64 `yaml:"end_time"`
}

type CutoffInfo struct {
	Reason       string `yaml:"reason"`
	EarlyBreak   bool   `yaml:"early_break"`
	NodesSkipped int    `yaml:"nodes_skipped"`
}

type JobStatistics struct {
	TotalNodes     int `yaml:"total_nodes"`
	MaxDepth       int `yaml:"max_depth"`
	Cutoffs        int `yaml:"cutoffs"`
	EndgamesSolved int `yaml:"endgames_solved"`
}

type App struct {
	app       *tview.Application
	tree      *tview.TreeView
	info      *tview.TextView
	status    *tview.TextView
	logData   []LogData
	searchBox *tview.InputField
	searching bool
	root      *tview.Flex
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: pegviewer <logfile>")
		os.Exit(1)
	}

	logFile := os.Args[1]
	
	app := &App{
		app: tview.NewApplication(),
	}

	if err := app.loadLogFile(logFile); err != nil {
		fmt.Printf("Error loading log file: %v\n", err)
		os.Exit(1)
	}

	app.setupUI()
	app.setupKeys()

	if err := app.app.Run(); err != nil {
		panic(err)
	}
}

func (a *App) loadLogFile(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return err
	}

	// Handle multiple YAML documents separated by ---
	content := string(data)
	
	// Split by --- separators
	docs := strings.Split(content, "---")
	
	for _, doc := range docs {
		doc = strings.TrimSpace(doc)
		if doc == "" {
			continue
		}
		
		var logData LogData
		if err := yaml.Unmarshal([]byte(doc), &logData); err != nil {
			fmt.Printf("Error parsing document: %v\n", err)
			continue
		}
		
		a.logData = append(a.logData, logData)
	}

	if len(a.logData) == 0 {
		return fmt.Errorf("no valid log data found in file")
	}

	return nil
}

func (a *App) setupUI() {
	// Create tree view
	a.tree = tview.NewTreeView()
	a.tree.SetTitle("PEG Log Viewer").SetBorder(true)
	a.tree.SetSelectedFunc(a.onNodeSelected)

	// Create info panel
	a.info = tview.NewTextView()
	a.info.SetTitle("Details").SetBorder(true)
	a.info.SetDynamicColors(true)
	a.info.SetScrollable(true)

	// Create status bar
	a.status = tview.NewTextView()
	a.status.SetText("Press 'q' to quit, '/' to search, 'Space' to expand/collapse, '?' for help")
	a.status.SetTextAlign(tview.AlignCenter)

	// Create search box (initially hidden)
	a.searchBox = tview.NewInputField()
	a.searchBox.SetLabel("Search: ")
	a.searchBox.SetFieldWidth(50)
	a.searchBox.SetDoneFunc(a.onSearchDone)
	a.searchBox.SetChangedFunc(a.onSearchChanged)

	// Build the tree
	a.buildTree()

	// Layout
	main := tview.NewFlex().
		AddItem(a.tree, 0, 2, true).
		AddItem(a.info, 0, 1, false)

	a.root = tview.NewFlex().SetDirection(tview.FlexRow).
		AddItem(main, 0, 1, true).
		AddItem(a.status, 1, 1, false)

	a.app.SetRoot(a.root, true)
}

func (a *App) buildTree() {
	root := tview.NewTreeNode("PEG Logs").
		SetColor(tcell.ColorYellow).
		SetExpanded(true)

	for i, logData := range a.logData {
		jobNode := tview.NewTreeNode(fmt.Sprintf("Job %d: %s", i+1, logData.PEGPlay)).
			SetColor(tcell.ColorGreen).
			SetExpanded(false).
			SetReference(&logData)

		// Add metadata
		metaNode := tview.NewTreeNode(fmt.Sprintf("Meta (Thread: %d, Time: %s)", 
			logData.Meta.Thread, logData.Meta.Timestamp)).
			SetColor(tcell.ColorBlue).
			SetExpanded(false)
		jobNode.AddChild(metaNode)

		// Add statistics
		statsNode := tview.NewTreeNode(fmt.Sprintf("Stats (Nodes: %d, Depth: %d, Cutoffs: %d)", 
			logData.Statistics.TotalNodes, logData.Statistics.MaxDepth, logData.Statistics.Cutoffs)).
			SetColor(tcell.ColorBlue).
			SetExpanded(false)
		jobNode.AddChild(statsNode)

		// Add options
		for j, option := range logData.Options {
			optionNode := tview.NewTreeNode(fmt.Sprintf("Option %d: %s (%d perms)", 
				j+1, option.PermutationInBag, option.PermutationCount)).
				SetColor(tcell.ColorWhite).
				SetExpanded(false).
				SetReference(&option)

			// Add execution tree
			if option.ExecutionTree.Root.Move != "" {
				treeNode := tview.NewTreeNode("Execution Tree").
					SetColor(tcell.ColorPurple).
					SetExpanded(false)
				
				a.addTreeNode(treeNode, &option.ExecutionTree.Root)
				optionNode.AddChild(treeNode)
			}

			jobNode.AddChild(optionNode)
		}

		root.AddChild(jobNode)
	}

	a.tree.SetRoot(root).SetCurrentNode(root)
}

func (a *App) addTreeNode(parent *tview.TreeNode, node *TreeNode) {
	// Format node text with timing and outcome info
	text := fmt.Sprintf("D%d P%d: %s", node.Depth, node.Player, node.Move)
	
	// Add timing info if available
	if node.Timing.DurationMs > 0 {
		text += fmt.Sprintf(" [%dms]", node.Timing.DurationMs)
	}
	
	// Add outcome info
	if node.GameState.OutcomeAfter != "" {
		text += fmt.Sprintf(" → %s", node.GameState.OutcomeAfter)
	}

	// Color based on depth and outcome
	color := tcell.ColorWhite
	if node.Cutoff.EarlyBreak {
		color = tcell.ColorRed
	} else if node.GameState.OutcomeAfter == "win" {
		color = tcell.ColorGreen
	} else if node.GameState.OutcomeAfter == "loss" {
		color = tcell.ColorRed
	}

	childNode := tview.NewTreeNode(text).
		SetColor(color).
		SetExpanded(false).
		SetReference(node)

	parent.AddChild(childNode)

	// Add children recursively
	for _, child := range node.Children {
		a.addTreeNode(childNode, &child)
	}
}

func (a *App) onNodeSelected(node *tview.TreeNode) {
	ref := node.GetReference()
	if ref == nil {
		a.info.SetText("No details available")
		return
	}

	var details strings.Builder
	
	switch v := ref.(type) {
	case *LogData:
		details.WriteString(fmt.Sprintf("[yellow]Job: %s[white]\n\n", v.PEGPlay))
		details.WriteString(fmt.Sprintf("Thread: %d\n", v.Meta.Thread))
		details.WriteString(fmt.Sprintf("Timestamp: %s\n", v.Meta.Timestamp))
		details.WriteString(fmt.Sprintf("Bag State: %s\n", v.Meta.BagState))
		details.WriteString(fmt.Sprintf("Empties Bag: %v\n", v.Meta.EmptiesBag))
		details.WriteString(fmt.Sprintf("Found Losses: %d\n", v.FoundLosses))
		details.WriteString(fmt.Sprintf("Min Potential Losses: %d\n", v.MinPotentialLosses))
		details.WriteString(fmt.Sprintf("Endgame Plies: %d\n", v.EndgamePlies))
		details.WriteString(fmt.Sprintf("\n[green]Statistics:[white]\n"))
		details.WriteString(fmt.Sprintf("Total Nodes: %d\n", v.Statistics.TotalNodes))
		details.WriteString(fmt.Sprintf("Max Depth: %d\n", v.Statistics.MaxDepth))
		details.WriteString(fmt.Sprintf("Cutoffs: %d\n", v.Statistics.Cutoffs))
		details.WriteString(fmt.Sprintf("Endgames Solved: %d\n", v.Statistics.EndgamesSolved))
		
	case *JobOption:
		details.WriteString(fmt.Sprintf("[yellow]Option: %s[white]\n\n", v.PermutationInBag))
		details.WriteString(fmt.Sprintf("Permutation Count: %d\n", v.PermutationCount))
		details.WriteString(fmt.Sprintf("Opponent Rack: %s\n", v.OppRack))
		details.WriteString(fmt.Sprintf("Our Rack: %s\n", v.OurRack))
		details.WriteString(fmt.Sprintf("Final Spread: %d\n", v.FinalSpread))
		details.WriteString(fmt.Sprintf("Time to Solve: %dms\n", v.TimeToSolveMs))
		
	case *TreeNode:
		details.WriteString(fmt.Sprintf("[yellow]Move: %s[white]\n\n", v.Move))
		details.WriteString(fmt.Sprintf("Depth: %d\n", v.Depth))
		details.WriteString(fmt.Sprintf("Player: %d\n", v.Player))
		details.WriteString(fmt.Sprintf("Duration: %dms\n", v.Timing.DurationMs))
		details.WriteString(fmt.Sprintf("\n[green]Game State:[white]\n"))
		details.WriteString(fmt.Sprintf("Spread: %d\n", v.GameState.Spread))
		details.WriteString(fmt.Sprintf("Outcome Before: %s\n", v.GameState.OutcomeBefore))
		details.WriteString(fmt.Sprintf("Outcome After: %s\n", v.GameState.OutcomeAfter))
		details.WriteString(fmt.Sprintf("Bag Remaining: %d\n", v.GameState.BagRemaining))
		details.WriteString(fmt.Sprintf("Game Over: %v\n", v.GameState.GameOver))
		
		if v.Cutoff.EarlyBreak {
			details.WriteString(fmt.Sprintf("\n[red]Cutoff:[white]\n"))
			details.WriteString(fmt.Sprintf("Reason: %s\n", v.Cutoff.Reason))
			details.WriteString(fmt.Sprintf("Nodes Skipped: %d\n", v.Cutoff.NodesSkipped))
		}
		
		if len(v.Children) > 0 {
			details.WriteString(fmt.Sprintf("\nChildren: %d\n", len(v.Children)))
		}
	}

	a.info.SetText(details.String())
}

func (a *App) setupKeys() {
	a.app.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		if a.searching {
			return event // Let search box handle input
		}
		
		switch event.Key() {
		case tcell.KeyRune:
			switch event.Rune() {
			case 'q':
				a.app.Stop()
				return nil
			case '/':
				a.startSearch()
				return nil
			case '?':
				a.showHelp()
				return nil
			case ' ': // Space to toggle expand/collapse
				node := a.tree.GetCurrentNode()
				if node != nil {
					node.SetExpanded(!node.IsExpanded())
				}
				return nil
			case 'j':
				a.tree.InputHandler()(tcell.NewEventKey(tcell.KeyDown, 0, tcell.ModNone), nil)
				return nil
			case 'k':
				a.tree.InputHandler()(tcell.NewEventKey(tcell.KeyUp, 0, tcell.ModNone), nil)
				return nil
			case 'l':
				node := a.tree.GetCurrentNode()
				if node != nil {
					node.SetExpanded(true)
				}
				return nil
			case 'h':
				node := a.tree.GetCurrentNode()
				if node != nil {
					node.SetExpanded(false)
				}
				return nil
			}
		case tcell.KeyEscape:
			if a.searching {
				a.onSearchDone(tcell.KeyEscape)
			}
			return nil
		}
		return event
	})
}

func (a *App) startSearch() {
	a.searching = true
	a.searchBox.SetText("")
	
	// Replace status with search box
	a.root.RemoveItem(a.status)
	a.root.AddItem(a.searchBox, 3, 1, false)
	
	a.app.SetFocus(a.searchBox)
}

func (a *App) onSearchDone(key tcell.Key) {
	a.searching = false
	
	// If search was cancelled, reset colors
	if key == tcell.KeyEscape {
		a.resetTreeColors(a.tree.GetRoot())
	}
	
	// Replace search box with status
	a.root.RemoveItem(a.searchBox)
	a.root.AddItem(a.status, 1, 1, false)
	
	a.app.SetFocus(a.tree)
}

func (a *App) onSearchChanged(text string) {
	// Reset all colors first
	a.resetTreeColors(a.tree.GetRoot())
	
	if text == "" {
		return
	}
	
	// Search through tree nodes
	a.searchInTree(a.tree.GetRoot(), strings.ToLower(text))
}

func (a *App) resetTreeColors(node *tview.TreeNode) {
	// Reset color based on node type
	ref := node.GetReference()
	if ref != nil {
		switch v := ref.(type) {
		case *LogData:
			node.SetColor(tcell.ColorGreen)
		case *JobOption:
			node.SetColor(tcell.ColorWhite)
		case *TreeNode:
			if v.Cutoff.EarlyBreak {
				node.SetColor(tcell.ColorRed)
			} else if v.GameState.OutcomeAfter == "win" {
				node.SetColor(tcell.ColorGreen)
			} else if v.GameState.OutcomeAfter == "loss" {
				node.SetColor(tcell.ColorRed)
			} else {
				node.SetColor(tcell.ColorWhite)
			}
		}
	} else {
		// Default colors for special nodes
		text := node.GetText()
		if strings.Contains(text, "PEG Logs") {
			node.SetColor(tcell.ColorYellow)
		} else if strings.Contains(text, "Meta") || strings.Contains(text, "Stats") {
			node.SetColor(tcell.ColorBlue)
		} else if strings.Contains(text, "Execution Tree") {
			node.SetColor(tcell.ColorPurple)
		} else {
			node.SetColor(tcell.ColorWhite)
		}
	}
	
	// Reset children
	for _, child := range node.GetChildren() {
		a.resetTreeColors(child)
	}
}

func (a *App) searchInTree(node *tview.TreeNode, searchText string) bool {
	nodeText := strings.ToLower(node.GetText())
	found := strings.Contains(nodeText, searchText)
	
	// Search children
	childFound := false
	for _, child := range node.GetChildren() {
		if a.searchInTree(child, searchText) {
			childFound = true
			node.SetExpanded(true) // Expand parent if child matches
		}
	}
	
	// If this node or any child matches, highlight this node
	if found || childFound {
		if found {
			node.SetColor(tcell.ColorYellow)
		}
		return true
	}
	
	return false
}

func (a *App) showHelp() {
	helpText := `[yellow]PEG Log Viewer Help[white]

Navigation:
  ↑/↓ or j/k    Move up/down
  →/l           Expand node
  ←/h           Collapse node
  Space         Toggle expand/collapse
  
Search:
  /             Start search
  Enter/Esc     End search
  
Other:
  q             Quit
  ?             Show this help
  
Colors:
  [green]Green[white]   - Winning outcomes / Job nodes
  [red]Red[white]     - Losing outcomes / Early cutoffs
  [yellow]Yellow[white]  - Search matches / Root node
  [blue]Blue[white]    - Metadata
  [purple]Purple[white] - Execution trees
`
	
	modal := tview.NewModal().
		SetText(helpText).
		AddButtons([]string{"OK"}).
		SetDoneFunc(func(buttonIndex int, buttonLabel string) {
			a.app.SetRoot(a.root, true)
		})
	
	a.app.SetRoot(modal, true)
}