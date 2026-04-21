package shell

import (
	"fmt"
	"strconv"
	"strings"
)

// OptType is the value shape of an Option.
type OptType int

const (
	OptString OptType = iota
	OptInt
	OptBool
	OptStringArray
)

// Option declares one flag accepted by a command.
type Option struct {
	Name       string
	Type       OptType
	Default    any
	Help       string
	Values     []string
	ValuesFunc func(sc *ShellController) []string
	// Deprecated, if non-empty, marks this option as removed. validateSpecOptions
	// returns an error containing this string as the suggested replacement.
	Deprecated string
}

// CommandSpec declares the shape of one shell command. Registering a spec
// opts the command into option-name validation, type checking, and
// autocomplete driven by the same declaration.
type CommandSpec struct {
	Name     string
	Options  []Option
	Verbs    []string
	ArgsFunc func(sc *ShellController) []string
}

var commandSpecs = map[string]*CommandSpec{}

func registerSpec(s *CommandSpec) {
	if _, exists := commandSpecs[s.Name]; exists {
		panic("duplicate command spec: " + s.Name)
	}
	commandSpecs[s.Name] = s
}

func (s *CommandSpec) lookupOption(name string) *Option {
	for i := range s.Options {
		if s.Options[i].Name == name {
			return &s.Options[i]
		}
	}
	return nil
}

// validateSpecOptions rejects unknown options and malformed values against
// the registered CommandSpec. Commands without a spec are not validated —
// migration is opt-in.
func validateSpecOptions(cmd *shellcmd) error {
	spec, ok := commandSpecs[cmd.cmd]
	if !ok {
		return nil
	}
	for key, vals := range cmd.options {
		opt := spec.lookupOption(key)
		if opt == nil {
			return fmt.Errorf("unknown option -%s for command '%s'", key, cmd.cmd)
		}
		if opt.Deprecated != "" {
			return fmt.Errorf("-%s has been removed; use %s instead", key, opt.Deprecated)
		}
		if err := validateOptValues(opt, vals); err != nil {
			return err
		}
	}
	return nil
}

// specOptionsFor returns a flat slice of "-name" strings for use in
// autocomplete option-name completion. Legacy commandMetadata still holds
// commands that haven't been migrated.
// specOptionsFor returns tab-completion candidates for a command's options.
// Deprecated options are excluded.
func specOptionsFor(spec *CommandSpec) []string {
	out := make([]string, 0, len(spec.Options))
	for _, o := range spec.Options {
		if o.Deprecated == "" {
			out = append(out, "-"+o.Name)
		}
	}
	return out
}

// ---- Command specs ----
// All specs registered here; one per command. Order is alphabetical.

func init() {
	registerSpec(&CommandSpec{
		Name:  "peg",
		Verbs: []string{"stop", "output"},
		Options: []Option{
			{Name: "endgameplies", Type: OptInt, Default: 4,
				Help: "how many plies to search in each endgame"},
			{Name: "maxtime", Type: OptInt, Default: 0,
				Help: "max solve time in seconds; 0 = unlimited"},
			{Name: "threads", Type: OptInt, Default: 0,
				Help: "thread count; 0 = all cores"},
			{Name: "maxsolutions", Type: OptInt, Default: 3,
				Help: "max number of solutions to return"},
			{Name: "max-tiles-left", Type: OptInt, Default: 1,
				Help: "only analyze plays that leave at most N tiles in the bag (-1 = all)"},
			{Name: "max-nested-depth", Type: OptInt, Default: 1,
				Help: "cap on nested sub-PEG depth (-1 = unlimited)"},
			{Name: "opprack", Type: OptString,
				Help: "known partial opponent rack"},
			{Name: "only-solve", Type: OptStringArray,
				Help: "only solve these specific plays (repeatable)"},
			{Name: "avoid-prune", Type: OptStringArray,
				Help: "never prune these plays under early-cutoff"},
			{Name: "skip-loss", Type: OptBool, Values: boolValues,
				Help: "skip each play as soon as a losing outcome is found"},
			{Name: "early-cutoff", Type: OptBool, Values: boolValues,
				Help: "prune plays known to be worse than the best found"},
			{Name: "skip-tiebreaker", Type: OptBool, Values: boolValues,
				Help: "skip spread-based tiebreaker among equal-win plays"},
			{Name: "disable-id", Type: OptBool, Values: boolValues,
				Help: "disable iterative deepening"},
			{Name: "skip-deep-pass", Type: OptBool, Default: true, Values: boolValues,
				Help: "suppress pass as a candidate inside nested sub-PEGs"},
			{Name: "log", Type: OptBool, Values: boolValues,
				Help: "write per-thread solve logs"},
			{Name: "skip-non-emptying", Deprecated: "-max-tiles-left 0"},
		},
	})

	registerSpec(&CommandSpec{
		Name:  "endgame",
		Verbs: []string{"stop", "metrics", "output"},
		Options: []Option{
			{Name: "plies", Type: OptInt, Default: 4},
			{Name: "maxtime", Type: OptInt, Default: 0},
			{Name: "threads", Type: OptInt, Default: 0},
			{Name: "multiple-vars", Type: OptInt, Default: 1},
			{Name: "parallel-algo", Type: OptString},
			{Name: "disable-id", Type: OptBool, Values: boolValues},
			{Name: "disable-tt", Type: OptBool, Values: boolValues},
			{Name: "first-win-optim", Type: OptBool, Values: boolValues},
			{Name: "prevent-slowroll", Type: OptBool, Values: boolValues},
			{Name: "disable-negascout", Type: OptBool, Values: boolValues},
			{Name: "null-window", Type: OptBool, Values: boolValues},
			{Name: "log", Type: OptBool, Values: boolValues},
			{Name: "also-solve-var", Type: OptString},
		},
	})

	registerSpec(&CommandSpec{
		Name:  "sim",
		Verbs: []string{"stop", "continue", "show", "details", "log", "trim", "heatmap", "playstats", "tilestats"},
		Options: []Option{
			{Name: "plies", Type: OptInt},
			{Name: "threads", Type: OptInt},
			{Name: "stop", Type: OptInt, Values: stopValues},
			{Name: "opprack", Type: OptString},
			{Name: "useinferences", Type: OptString,
				Values: []string{"weightedrandomtiles", "weightedrandomracks"}},
			{Name: "collect-heatmap", Type: OptBool, Values: boolValues},
			{Name: "fixedsimiters", Type: OptInt},
			{Name: "fixedsimplies", Type: OptInt},
			{Name: "fixedsimcount", Type: OptInt},
			{Name: "autostopcheckinterval", Type: OptInt},
			{Name: "stop-ppscaling", Type: OptInt},
			{Name: "stop-itercutoff", Type: OptInt},
			{Name: "avoid-prune", Type: OptStringArray},
		},
	})

	registerSpec(&CommandSpec{
		Name:  "infer",
		Verbs: []string{"log", "details", "output"},
		Options: []Option{
			{Name: "threads", Type: OptInt},
			{Name: "time", Type: OptInt},
		},
	})

	registerSpec(&CommandSpec{
		Name: "render",
		Options: []Option{
			{Name: "tile-color", Type: OptString,
				Values: []string{"orange", "yellow", "pink", "red", "blue", "black", "white"}},
			{Name: "board-color", Type: OptString,
				Values: []string{"jade", "teal", "blue", "purple", "green", "darkgreen", "brown"}},
			{Name: "heatmap", Type: OptString},
			{Name: "ply", Type: OptString},
		},
	})

	registerSpec(&CommandSpec{
		Name: "commit",
		Options: []Option{
			{Name: "tileorder", Type: OptString},
		},
	})

	registerSpec(&CommandSpec{
		Name: "analyze",
		Options: []Option{
			{Name: "json", Type: OptString},
			{Name: "force", Type: OptBool, Values: boolValues},
			{Name: "player", Type: OptString},
		},
	})

	registerSpec(&CommandSpec{
		Name: "analyze-batch",
		Options: []Option{
			{Name: "continue", Type: OptBool, Values: boolValues},
			{Name: "summary-only", Type: OptBool, Values: boolValues},
			{Name: "batch", Type: OptString},
			{Name: "force", Type: OptBool, Values: boolValues},
			{Name: "json", Type: OptString},
			{Name: "player", Type: OptString},
		},
	})

	registerSpec(&CommandSpec{
		Name: "analyze-view",
		Options: []Option{
			{Name: "batch", Type: OptString},
		},
	})

	registerSpec(&CommandSpec{
		Name: "analyze-browse",
		Options: []Option{
			{Name: "delete", Type: OptString},
			{Name: "yes", Type: OptBool, Values: boolValues},
			{Name: "batch", Type: OptString},
			{Name: "limit", Type: OptInt},
		},
	})

	// autoanalyze: export a game to GCG, or analyze a log file.
	// Note: commandMetadata previously listed -count and -single-turn-only
	// which were never read by the handler. Correct options are below.
	registerSpec(&CommandSpec{
		Name: "autoanalyze",
		Options: []Option{
			{Name: "export", Type: OptString},
			{Name: "letterdist", Type: OptString},
			{Name: "lex", Type: OptString},
			{Name: "boardlayout", Type: OptString},
		},
	})

	registerSpec(&CommandSpec{
		Name: "autoplay",
		Options: []Option{
			{Name: "botcode1", Type: OptString, Values: botCodes},
			{Name: "botcode2", Type: OptString, Values: botCodes},
			{Name: "threads", Type: OptInt},
			{Name: "logfile", Type: OptString},
			{Name: "lexicon", Type: OptString},
			{Name: "letterdistribution", Type: OptString},
			{Name: "leavefile1", Type: OptString},
			{Name: "leavefile2", Type: OptString},
			{Name: "pegfile1", Type: OptString},
			{Name: "pegfile2", Type: OptString},
			{Name: "minsimplies1", Type: OptInt},
			{Name: "minsimplies2", Type: OptInt},
			{Name: "numgames", Type: OptInt},
			{Name: "stochastic1", Type: OptBool, Values: boolValues},
			{Name: "stochastic2", Type: OptBool, Values: boolValues},
			{Name: "block", Type: OptBool, Values: boolValues},
			{Name: "genseeds", Type: OptBool, Values: boolValues},
			{Name: "deterministic", Type: OptBool, Values: boolValues},
			{Name: "seedfile", Type: OptString},
		},
	})

	// speedtest accepts sim-style options.
	registerSpec(&CommandSpec{
		Name: "speedtest",
		Options: []Option{
			{Name: "plies", Type: OptInt},
			{Name: "threads", Type: OptInt},
			{Name: "stop", Type: OptInt, Values: stopValues},
			{Name: "opprack", Type: OptString},
		},
	})

	// explain accepts sim-style options (plies/stop passed through to simmer).
	registerSpec(&CommandSpec{
		Name: "explain",
		Options: []Option{
			{Name: "plies", Type: OptInt},
			{Name: "stop", Type: OptInt, Values: stopValues},
			{Name: "threads", Type: OptInt},
			{Name: "opprack", Type: OptString},
		},
	})

	// gen has no options (the -equity metadata entry was dead code).
	registerSpec(&CommandSpec{Name: "gen"})

	// args-only / verb-only commands
	registerSpec(&CommandSpec{
		Name: "set",
		ArgsFunc: func(_ *ShellController) []string {
			return []string{"lexicon", "challenge", "variation", "board", "lowercase"}
		},
	})
	registerSpec(&CommandSpec{
		Name: "setconfig",
		ArgsFunc: func(_ *ShellController) []string {
			return []string{
				"data-path", "default-lexicon", "default-letter-distribution",
				"triton-use-triton", "triton-url", "triton-model-name", "triton-model-version",
			}
		},
	})
	registerSpec(&CommandSpec{
		Name: "alias",
		ArgsFunc: func(_ *ShellController) []string {
			return []string{"set", "delete", "show", "list", "remove", "rm"}
		},
	})
	registerSpec(&CommandSpec{
		Name: "mode",
		ArgsFunc: func(_ *ShellController) []string {
			return []string{"standard", "endgamedebug"}
		},
	})
	varIDsFunc := func(sc *ShellController) []string {
		if sc.currentVariation == nil {
			return nil
		}
		node := sc.currentVariation
		for node != nil && len(node.children) <= 1 {
			node = node.parent
		}
		if node == nil || len(node.children) <= 1 {
			return nil
		}
		var ids []string
		for _, child := range node.children {
			if child.variationID == 0 {
				ids = append(ids, "main")
			} else {
				ids = append(ids, strconv.Itoa(child.variationID))
			}
		}
		return ids
	}
	registerSpec(&CommandSpec{
		Name:     "var",
		Verbs:    []string{"list", "main", "info", "delete", "promote"},
		ArgsFunc: varIDsFunc,
	})
	registerSpec(&CommandSpec{
		Name:     "variation",
		Verbs:    []string{"list", "main", "info", "delete", "promote"},
		ArgsFunc: varIDsFunc,
	})

	// Leaf commands — no options, no special args. Registered for
	// commandNames coverage and future validation completeness.
	for _, name := range []string{
		"exit", "help", "new", "load", "unload", "last", "n", "p", "s",
		"name", "note", "turn", "rack", "add", "challenge", "aiplay",
		"hastyplay", "selftest", "list", "export", "script",
		"analyze-turn", "volunteer", "gid", "leave", "cgp", "check",
		"update", "gamestate", "mleval", "winpct", "build-wmp",
	} {
		registerSpec(&CommandSpec{Name: name})
	}
}

func validateOptValues(opt *Option, vals []string) error {
	if len(vals) == 0 {
		return nil
	}
	switch opt.Type {
	case OptStringArray, OptString:
		return nil
	case OptInt:
		if len(vals) > 1 {
			return fmt.Errorf("option -%s: expected single value, got %d", opt.Name, len(vals))
		}
		if _, err := strconv.Atoi(vals[0]); err != nil {
			return fmt.Errorf("option -%s: invalid int %q", opt.Name, vals[0])
		}
	case OptBool:
		if len(vals) > 1 {
			return fmt.Errorf("option -%s: expected single value, got %d", opt.Name, len(vals))
		}
		s := strings.ToLower(vals[0])
		if s != "true" && s != "false" {
			return fmt.Errorf("option -%s: invalid bool %q (use true or false)", opt.Name, vals[0])
		}
	}
	return nil
}
