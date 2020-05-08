package shell

import (
	"testing"

	"github.com/matryer/is"
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
			&shellcmd{"autoplay", nil, map[string]string{"file": "/path/to/log.txt"}},
			nil},
		{"autoplay stop",
			&shellcmd{"autoplay", []string{"stop"}, map[string]string{}},
			nil},
		{"autoplay exhaustiveleave noleave -file foo.txt ",
			&shellcmd{"autoplay",
				[]string{"exhaustiveleave", "noleave"},
				map[string]string{"file": "foo.txt"}},
			nil,
		},
		{"autoplay exhaustiveleave noleave -file",
			nil, errWrongOptionSyntax},
	}
	for _, t := range cases {
		cmd, err := extractFields(t.line)
		is.Equal(cmd, t.expCmd)
		is.Equal(err, t.expErr)
	}
}
