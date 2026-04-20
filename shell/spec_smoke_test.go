package shell

import (
	"strings"
	"testing"
)

func TestPegSpecValidation(t *testing.T) {
	tests := []struct {
		line        string
		wantErr     bool
		errContains string
	}{
		{"peg -threads 4", false, ""},
		{"peg -max-tiles-left 0", false, ""},
		{"peg -skip-deep-pass true", false, ""},
		{"peg -skip-deep-pass false", false, ""},
		{"peg -only-solve X -only-solve Y", false, ""},
		{"peg -endgameplies abc", true, "invalid int"},
		{"peg -skip-loss maybe", true, "invalid bool"},
		{"peg -skip-non-emptyers true", true, "unknown option"},
		{"peg -skip-non-emptying true", true, "-max-tiles-left 0"}, // deprecated → helpful message
		{"peg stop", false, ""},
	}
	for _, tt := range tests {
		cmd, err := extractFields(tt.line)
		if err != nil {
			t.Fatalf("parse %q: %v", tt.line, err)
		}
		err = validateSpecOptions(cmd)
		if (err != nil) != tt.wantErr {
			t.Errorf("validateSpecOptions(%q) err=%v, wantErr=%v", tt.line, err, tt.wantErr)
			continue
		}
		if tt.errContains != "" && (err == nil || !strings.Contains(err.Error(), tt.errContains)) {
			t.Errorf("validateSpecOptions(%q) err=%v, want it to contain %q", tt.line, err, tt.errContains)
		}
	}
}
