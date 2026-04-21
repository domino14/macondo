package shell

import (
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

// allCommandNames returns all registered command names plus aliases.
func allCommandNames(aliases map[string]string) []string {
	names := make([]string, 0, len(commandSpecs)+len(aliases))
	for name := range commandSpecs {
		names = append(names, name)
	}
	for name := range aliases {
		names = append(names, name)
	}
	return names
}

// Do implements the readline.AutoComplete interface.
func (c *ShellCompleter) Do(line []rune, pos int) ([][]rune, int) {
	text := string(line[:pos])

	fields, err := shellquote.Split(text)
	if err != nil {
		fields = strings.Fields(text)
	}

	endsWithSpace := len(text) > 0 && text[len(text)-1] == ' '

	var prefix string
	var completions []string

	if len(fields) == 0 || (len(fields) == 1 && !endsWithSpace) {
		// Completing a command name.
		if len(fields) == 1 {
			prefix = fields[0]
		}
		completions = allCommandNames(c.sc.aliases)
	} else {
		cmdName := fields[0]

		// Expand alias to get real command name.
		if aliasValue, isAlias := c.sc.aliases[cmdName]; isAlias {
			aliasFields, err := shellquote.Split(aliasValue)
			if err == nil && len(aliasFields) > 0 {
				cmdName = aliasFields[0]
			}
		}

		if !endsWithSpace && len(fields) > 0 {
			prefix = fields[len(fields)-1]
		}

		// Identify the last fully-typed token for value-completion context.
		var lastCompleteField string
		if endsWithSpace && len(fields) > 0 {
			lastCompleteField = fields[len(fields)-1]
		} else if len(fields) > 1 {
			lastCompleteField = fields[len(fields)-2]
		}

		// Value completion: after "-optname <space>", offer the option's values.
		if lastCompleteField != "" && strings.HasPrefix(lastCompleteField, "-") {
			optName := strings.TrimPrefix(lastCompleteField, "-")
			if spec, ok := commandSpecs[cmdName]; ok {
				if opt := spec.lookupOption(optName); opt != nil {
					if opt.ValuesFunc != nil {
						completions = opt.ValuesFunc(c.sc)
					} else {
						completions = opt.Values // nil = free-form
					}
				}
			}
		}

		// Option-name and verb/arg completion.
		if completions == nil {
			if spec, ok := commandSpecs[cmdName]; ok {
				if strings.HasPrefix(prefix, "-") {
					completions = specOptionsFor(spec)
				} else {
					completions = append([]string{}, spec.Verbs...)
					if spec.ArgsFunc != nil {
						completions = append(completions, spec.ArgsFunc(c.sc)...)
					}
				}
			}
		}
	}

	// Filter by prefix and return only the suffix to append.
	var matches [][]rune
	for _, completion := range completions {
		if strings.HasPrefix(completion, prefix) {
			matches = append(matches, []rune(completion[len(prefix):]))
		}
	}
	return matches, len(prefix)
}
