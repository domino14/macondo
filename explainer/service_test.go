package explainer

import (
	"context"
	"strings"
	"testing"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/domino14/macondo/config"
	"github.com/matryer/is"
)

// MACONDO_NO_LLM lets the whole service path run without an API key: the
// prompt comes back as the explanation. That covers everything except the
// call itself, including what gets kept for -show-previous-prompt.
func TestExplainKeepsTheLastExchange(t *testing.T) {
	if testing.Short() {
		t.Skip("runs a simulation")
	}
	is := is.New(t)
	t.Setenv("MACONDO_NO_LLM", "1")

	an, simmer, simStats := simulate(t, examplePosition, 8, 3, 120)
	svc := NewService(config.DefaultConfig())

	is.Equal(svc.LastExchange(), (*Exchange)(nil)) // nothing to show yet

	all := simmer.CandidateStats()
	worst := all[len(all)-1]
	result, err := svc.Explain(context.Background(), &ExplainInput{
		Game:     an.game,
		Simmer:   simmer,
		SimStats: simStats,
		Compare:  &ComparisonRequest{Move: worst.Move, FromHistory: true},
	})
	is.NoErr(err)
	is.True(result.Prompt != nil)
	is.True(len(result.Concepts) > 0)

	last := svc.LastExchange()
	is.True(last != nil)
	is.Equal(last.Prompt, result.Prompt)
	is.Equal(last.Comparison, worst.Play)
	// Nothing was called, so what is kept says so rather than repeating the
	// prompt back as if the model had produced it.
	is.True(strings.Contains(last.Response, "no model was called"))

	// The dump identifies what it was about, so a prompt read back later is
	// recognizable.
	dump := last.String()
	is.True(strings.Contains(dump, "Explanation of "+last.BestPlay))
	is.True(strings.Contains(dump, "compared against "+worst.Play))
	is.True(strings.Contains(dump, "Concept cards selected"))

	// The sections have to appear in order, because which section a line
	// belongs to is decided by which banner precedes it - there is no
	// per-line marker that could tell them apart. The prompt is markdown, and
	// a heading like "## What to say" in system.md really is sent.
	order := []string{notesBanner, toolsBanner, systemBanner, userBanner, ResponseBanner}
	at := -1
	for _, banner := range order {
		i := strings.Index(dump, banner)
		is.True(i > at)
		at = i
	}

	// Our commentary lives entirely above the first message banner.
	notes, messages, found := strings.Cut(dump, systemBanner)
	is.True(found)
	is.True(strings.Contains(notes, "Concept cards selected"))
	is.True(!strings.Contains(messages, "Concept cards selected"))

	// The tool definitions are part of the request even though they are in
	// neither message, and their descriptions carry real instructions - so a
	// dump without them would be misleading about what the model was told.
	tools, _, found := strings.Cut(dump, systemBanner)
	is.True(found)
	for _, name := range []string{
		"get_our_play_metadata", "get_our_future_play_metadata", "evaluate_leave",
	} {
		is.True(strings.Contains(tools, `"name": "`+name+`"`))
	}
	is.True(strings.Contains(tools, "Only plays listed in the 'Our follow-up play' table"))

	// And the banners are not the kind of line markdown produces, so nothing
	// in the content can be mistaken for one.
	for _, banner := range order {
		is.Equal(strings.Count(dump, banner), 1)
	}
}

// Parameters() is a map, so the schema has to be rendered in a fixed order or
// two dumps of the same position won't diff.
func TestToolSchemasAreStable(t *testing.T) {
	is := is.New(t)
	an := NewAnalyzer()
	tools := []interfaces.Tool{
		NewGetOurPlayMetadataTool(an),
		NewGetOurFuturePlayMetadataTool(an),
		NewEvaluateLeaveTool(an),
	}

	first := ToolSchemas(tools)
	for range 5 {
		is.Equal(ToolSchemas(tools), first)
	}
	is.True(strings.Contains(first, `"required": [`))
	is.True(strings.Contains(first, `"type": "object"`))

	// No tools, no section - BuildPrompt's own callers don't set them.
	p := &Prompt{System: "sys", User: "usr"}
	is.True(!strings.Contains(p.String(), toolsBanner))
	p.Tools = tools
	is.True(strings.Contains(p.String(), toolsBanner))
}

func TestExplainRejectsAnEmptyInput(t *testing.T) {
	is := is.New(t)
	svc := NewService(config.DefaultConfig())

	_, err := svc.Explain(context.Background(), nil)
	is.True(err != nil)
	_, err = svc.Explain(context.Background(), &ExplainInput{})
	is.True(err != nil)
}
