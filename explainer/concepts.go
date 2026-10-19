package explainer

import (
	"embed"
	"fmt"
	"io/fs"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// The prompt used to explain every Scrabble concept on every call: what a
// pre-endgame is on a position with a full bag, how blank-variant rows are
// grouped when no blank is in play, how to read inference data that nothing
// ever supplies. All of it competed for the model's attention with the two or
// three ideas that actually applied.
//
// A concept is now a card with a trigger. The triggers are flags computed in
// facts.go, so which cards ship is a property of the position, decided in Go
// and testable without an API key.

//go:embed concepts/*.md
var conceptFS embed.FS

// Concept is one card: a piece of Scrabble knowledge plus the conditions under
// which it is worth spending prompt on.
type Concept struct {
	// ID is the card's name, taken from its frontmatter.
	ID string
	// Priority orders the cards in the prompt, lowest first.
	Priority int
	// When lists flag names. The card ships if any of them is set, so
	// "always" is just a flag that is always true.
	When []string
	Body string
}

var loadConcepts = sync.OnceValues(func() ([]*Concept, error) {
	entries, err := fs.ReadDir(conceptFS, "concepts")
	if err != nil {
		return nil, err
	}
	out := make([]*Concept, 0, len(entries))
	for _, e := range entries {
		b, err := conceptFS.ReadFile("concepts/" + e.Name())
		if err != nil {
			return nil, err
		}
		c, err := parseConcept(string(b))
		if err != nil {
			return nil, fmt.Errorf("concepts/%s: %w", e.Name(), err)
		}
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Priority != out[j].Priority {
			return out[i].Priority < out[j].Priority
		}
		return out[i].ID < out[j].ID
	})
	return out, nil
})

// parseConcept reads a card: a small frontmatter block, then the text.
//
//	---
//	id: setup
//	priority: 20
//	when: has_setup, has_opportunity
//	---
//	A setup play is ...
func parseConcept(s string) (*Concept, error) {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "---") {
		return nil, fmt.Errorf("missing frontmatter")
	}
	_, rest, found := strings.Cut(s, "---")
	if !found {
		return nil, fmt.Errorf("missing frontmatter")
	}
	front, body, found := strings.Cut(rest, "---")
	if !found {
		return nil, fmt.Errorf("unterminated frontmatter")
	}

	c := &Concept{Body: strings.TrimSpace(body)}
	for _, line := range strings.Split(front, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		key, value, found := strings.Cut(line, ":")
		if !found {
			return nil, fmt.Errorf("bad frontmatter line %q", line)
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		switch key {
		case "id":
			c.ID = value
		case "priority":
			p, err := strconv.Atoi(value)
			if err != nil {
				return nil, fmt.Errorf("bad priority %q", value)
			}
			c.Priority = p
		case "when":
			for _, flag := range strings.Split(value, ",") {
				if flag = strings.TrimSpace(flag); flag != "" {
					c.When = append(c.When, flag)
				}
			}
		default:
			return nil, fmt.Errorf("unknown frontmatter key %q", key)
		}
	}
	if c.ID == "" {
		return nil, fmt.Errorf("missing id")
	}
	if len(c.When) == 0 {
		return nil, fmt.Errorf("card %s has no trigger", c.ID)
	}
	if c.Body == "" {
		return nil, fmt.Errorf("card %s has no body", c.ID)
	}
	return c, nil
}

// SelectConcepts returns the cards this position calls for, in prompt order.
func SelectConcepts(fl Flags) ([]*Concept, error) {
	all, err := loadConcepts()
	if err != nil {
		return nil, err
	}
	out := []*Concept{}
	for _, c := range all {
		for _, flag := range c.When {
			if fl[flag] {
				out = append(out, c)
				break
			}
		}
	}
	return out, nil
}

// ConceptIDs names the cards, for tests and for logging which ones a position
// pulled in.
func ConceptIDs(cs []*Concept) []string {
	out := make([]string, 0, len(cs))
	for _, c := range cs {
		out = append(out, c.ID)
	}
	return out
}
