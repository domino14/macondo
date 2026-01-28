package shell

import (
	"strconv"
	"strings"

	"github.com/kballard/go-shellquote"
)

// ShellCompleter provides context-aware autocomplete for shell commands
type ShellCompleter struct {
	sc *ShellController
}

func NewShellCompleter(sc *ShellController) *ShellCompleter {
	return &ShellCompleter{sc: sc}
}

// CommandMetadata holds autocomplete information for a command
type CommandMetadata struct {
	Options []string // Available options for this command (e.g., "-plies", "-threads")
	Args    []string // Possible argument values (for non-option arguments)
}

// commandMetadata maps command names to their options and arguments
// These are extracted from the actual command implementations in api.go and sim.go
var commandMetadata = map[string]CommandMetadata{
	"sim": {
		Options: []string{
			"-plies", "-threads", "-stop", "-opprack", "-useinferences",
			"-collect-heatmap", "-fixedsimiters", "-fixedsimplies",
			"-fixedsimcount", "-autostopcheckinterval", "-stop-ppscaling",
			"-stop-itercutoff",
		},
	},
	"peg": {
		Options: []string{
			"-endgameplies", "-maxtime", "-threads", "-maxsolutions",
			"-opprack", "-skip-non-emptying", "-skip-loss", "-early-cutoff",
			"-skip-tiebreaker", "-disable-id", "-only-solve", "-log",
			"-avoid-prune",
		},
	},
	"endgame": {
		Options: []string{
			"-plies", "-maxtime", "-threads", "-multiple-vars",
			"-disable-id", "-disable-tt", "-first-win-optim",
			"-prevent-slowroll", "-disable-negascout", "-null-window",
			"-also-solve-var",
		},
	},
	"infer": {
		Options: []string{"-threads", "-time"},
	},
	"gen": {
		Options: []string{"-equity"},
	},
	"autoplay": {
		Options: []string{"-botcode1", "-botcode2", "-threads"},
	},
	"set": {
		Args: []string{"lexicon", "challenge", "variation", "board", "lowercase"},
	},
	"setconfig": {
		Args: []string{
			"data-path", "default-lexicon", "default-letter-distribution",
			"triton-use-triton", "triton-url", "triton-model-name",
			"triton-model-version",
		},
	},
	"analyze": {
		Options: []string{"-player"},
	},
	"autoanalyze": {
		Options: []string{"-count", "-single-turn-only"},
	},
	"alias": {
		Args: []string{"set", "delete", "show", "list", "remove", "rm"},
	},
	"mode": {
		Args: []string{"standard", "endgamedebug"},
	},
	"render": {
		Options: []string{"-tile-color", "-board-color", "-heatmap", "-ply"},
	},
	"var": {
		Args: []string{"list", "main", "info", "delete", "promote"},
	},
	"variation": {
		Args: []string{"list", "main", "info", "delete", "promote"},
	},
}

// Common command names for command completion
var commandNames = []string{
	"help", "alias", "new", "load", "unload", "last", "n", "p", "s",
	"name", "note", "turn", "rack", "set", "setconfig", "gen", "autoplay",
	"sim", "infer", "add", "challenge", "commit", "aiplay", "hastyplay",
	"selftest", "list", "endgame", "peg", "mode", "export", "render", "analyze",
	"autoanalyze", "script", "gid", "leave", "cgp", "check", "explain", "exit",
	"var", "variation",
}

// Common values for certain option types
var boolValues = []string{"true", "false"}
var stopValues = []string{"90", "95", "98", "99", "999"}

// Bot codes from macondo.proto BotRequest.BotCode enum
var botCodes = []string{
	"HASTY_BOT",
	"LEVEL1_COMMON_WORD_BOT",
	"LEVEL2_COMMON_WORD_BOT",
	"LEVEL3_COMMON_WORD_BOT",
	"LEVEL4_COMMON_WORD_BOT",
	"LEVEL1_PROBABILISTIC",
	"LEVEL2_PROBABILISTIC",
	"LEVEL3_PROBABILISTIC",
	"LEVEL4_PROBABILISTIC",
	"LEVEL5_PROBABILISTIC",
	"NO_LEAVE_BOT",
	"SIMMING_BOT",
	"HASTY_PLUS_ENDGAME_BOT",
	"SIMMING_INFER_BOT",
	"FAST_ML_BOT",
	"RANDOM_BOT_WITH_TEMPERATURE",
	"SIMMING_WITH_ML_EVAL_BOT",
	"CUSTOM_BOT",
}

// Do implements the readline.AutoComplete interface
// It provides context-aware autocomplete based on what's been typed
func (c *ShellCompleter) Do(line []rune, pos int) ([][]rune, int) {
	// Get the text up to the cursor position
	text := string(line[:pos])

	// Parse the line using shellquote to handle quoted strings properly
	fields, err := shellquote.Split(text)
	if err != nil {
		// If we can't parse, fall back to simple space splitting
		fields = strings.Fields(text)
	}

	// Check if we're in the middle of typing a word or just after a space
	endsWithSpace := len(text) > 0 && text[len(text)-1] == ' '

	// Determine what we're trying to complete
	var prefix string
	var completions []string

	if len(fields) == 0 || (len(fields) == 1 && !endsWithSpace) {
		// Completing a command name
		if len(fields) == 1 {
			prefix = fields[0]
		}
		completions = commandNames

		// Also include aliases
		for aliasName := range c.sc.aliases {
			completions = append(completions, aliasName)
		}
	} else {
		// We have a command, now complete its arguments/options
		cmdName := fields[0]

		// Check if this is an alias, and if so, expand it to get the real command
		if aliasValue, isAlias := c.sc.aliases[cmdName]; isAlias {
			aliasFields, err := shellquote.Split(aliasValue)
			if err == nil && len(aliasFields) > 0 {
				cmdName = aliasFields[0]
			}
		}

		if !endsWithSpace && len(fields) > 0 {
			prefix = fields[len(fields)-1]
		}

		// Get the last complete field to check context
		var lastCompleteField string
		if endsWithSpace && len(fields) > 0 {
			lastCompleteField = fields[len(fields)-1]
		} else if len(fields) > 1 {
			lastCompleteField = fields[len(fields)-2]
		}

		// Check if the last field was an option that expects specific values
		if lastCompleteField != "" && strings.HasPrefix(lastCompleteField, "-") {
			optName := strings.TrimPrefix(lastCompleteField, "-")

			// Provide context-specific completions based on option name
			switch optName {
			case "stop":
				completions = stopValues
			case "botcode1", "botcode2":
				completions = botCodes
			case "collect-heatmap", "skip-non-emptying", "skip-loss",
				"early-cutoff", "skip-tiebreaker", "disable-id",
				"disable-tt", "first-win-optim", "prevent-slowroll",
				"disable-negascout", "null-window", "single-turn-only",
				"log":
				completions = boolValues
			case "useinferences":
				completions = []string{"weightedrandomtiles", "weightedrandomracks"}
			case "tile-color":
				completions = []string{"orange", "yellow", "pink", "red", "blue", "black", "white"}
			case "board-color":
				completions = []string{"jade", "teal", "blue", "purple", "green", "darkgreen", "brown"}
			}
		}

		// Special handling for var/variation command to suggest variation IDs
		if (cmdName == "var" || cmdName == "variation") && completions == nil {
			// Find the branching point to get available variations
			if c.sc.currentVariation != nil {
				node := c.sc.currentVariation
				for node != nil && len(node.children) <= 1 {
					node = node.parent
				}
				if node != nil && len(node.children) > 1 {
					// Add variation IDs as suggestions
					var varIDs []string
					for _, child := range node.children {
						if child.variationID == 0 {
							varIDs = append(varIDs, "main")
						} else {
							varIDs = append(varIDs, strconv.Itoa(child.variationID))
						}
					}
					// Combine with static args
					if metadata, exists := commandMetadata[cmdName]; exists {
						completions = append(varIDs, metadata.Args...)
					} else {
						completions = varIDs
					}
				}
			}
		}

		// If we haven't determined completions yet, show command options/args
		if completions == nil {
			if metadata, exists := commandMetadata[cmdName]; exists {
				// If we're typing something that starts with -, show options
				if strings.HasPrefix(prefix, "-") {
					completions = metadata.Options
				} else {
					// Show args if available, otherwise show options
					if len(metadata.Args) > 0 {
						completions = metadata.Args
					} else {
						completions = metadata.Options
					}
				}
			}
		}
	}

	// Filter completions based on prefix
	var matches [][]rune
	for _, completion := range completions {
		if strings.HasPrefix(completion, prefix) {
			// Return only the part that needs to be added
			suffix := completion[len(prefix):]
			matches = append(matches, []rune(suffix))
		}
	}

	return matches, len(prefix)
}
