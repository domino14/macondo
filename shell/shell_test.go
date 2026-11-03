package shell

import (
	"testing"

	"github.com/matryer/is"

	pb "github.com/domino14/macondo/gen/api/proto/macondo"
)

func TestExtractFields(t *testing.T) {
	is := is.New(t)
	type testdata struct {
		line   string
		expCmd *shellcmd
		expErr error
	}
	cases := []testdata{
		{"", nil, errNoData},
		{"autoplay -file /path/to/log.txt",
			&shellcmd{"autoplay", nil, map[string][]string{"file": {"/path/to/log.txt"}}},
			nil},
		{"autoplay stop",
			&shellcmd{"autoplay", []string{"stop"}, map[string][]string{}},
			nil},
		{"autoplay exhaustiveleave noleave -file foo.txt ",
			&shellcmd{"autoplay",
				[]string{"exhaustiveleave", "noleave"},
				map[string][]string{"file": {"foo.txt"}}},
			nil,
		},
		// {"autoplay exhaustiveleave noleave -file",
		// 	nil, errWrongOptionSyntax},

		// A declared boolean flag does not eat the argument after it...
		{"analyze-batch -continue /path/to/games",
			&shellcmd{"analyze-batch",
				[]string{"/path/to/games"},
				map[string][]string{"continue": {"true"}}},
			nil},
		{"analyze-batch -force -continue /path/to/games",
			&shellcmd{"analyze-batch",
				[]string{"/path/to/games"},
				map[string][]string{"force": {"true"}, "continue": {"true"}}},
			nil},
		// ...but still takes a value when one is written out.
		{"analyze-batch -continue false /path/to/games",
			&shellcmd{"analyze-batch",
				[]string{"/path/to/games"},
				map[string][]string{"continue": {"false"}}},
			nil},
		// Options that take a value are unaffected.
		{"analyze-batch -player \"Eric Smith\" /path/to/games",
			&shellcmd{"analyze-batch",
				[]string{"/path/to/games"},
				map[string][]string{"player": {"Eric Smith"}}},
			nil},
	}
	for _, t := range cases {
		cmd, err := extractFields(t.line)
		is.Equal(cmd, t.expCmd)
		is.Equal(err, t.expErr)
	}
}

func TestApplyPlayerOverrides(t *testing.T) {
	is := is.New(t)
	p := &pb.AutoplayPlayerConfig{}
	applyPlayerOverrides(p, CmdOptions{"leavefile1": {"a.klv2"}, "pegfile1": {"a.json"}, "stochastic1": {"true"}}, "1")
	is.Equal(p.LeaveFile, "a.klv2")
	is.Equal(p.PegFile, "a.json")
	is.True(p.StochasticStaticEval)
}
