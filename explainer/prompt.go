package explainer

import (
	_ "embed"
	"encoding/json"
	"sort"
	"strings"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
)

// The prompt files used to be read from disk with paths relative to the
// working directory, so `explain` only worked if the shell had been started
// from the repo root. They are embedded now, like shell/render_template.html.

//go:embed system.md
var systemPromptMD string

//go:embed quirky.md
var quirkyMD string

// Prompt is what gets sent: stable instructions in the system message, and
// everything that changes with the position in the user message. Splitting
// them this way is also what lets a provider cache the system half.
type Prompt struct {
	System string
	User   string
	// Concepts names the cards this position pulled in. Useful for logging
	// and for tests that assert a position asks for the right knowledge.
	Concepts []string
	// Tools are the function definitions sent alongside the messages. They
	// appear in neither message's text - the API carries them in a field of
	// their own - so a dump showing only the messages would be missing
	// instructions the model really does receive, and some of them are
	// substantive: get_our_future_play_metadata's description is what tells
	// the model it may only look up plays from the follow-up table.
	Tools []interfaces.Tool
}

// Banners dividing a dump into sections. Each one labels everything below it
// up to the next banner, which is the only workable rule: the prompt is
// markdown, so a per-line marker can't distinguish our commentary from the
// content - "## What to say" is a heading in system.md and really is sent.
//
// The eight-equals form is deliberate. A markdown setext heading is a line of
// nothing but "=", so a banner with text in it can't be mistaken for one in
// the model's reply.
const (
	bannerRule   = "========"
	notesBanner  = bannerRule + " MACONDO DEBUG NOTES (not sent to the model) " + bannerRule
	toolsBanner  = bannerRule + " TOOL DEFINITIONS (sent in a field of their own, not in the message text) " + bannerRule
	systemBanner = bannerRule + " SYSTEM MESSAGE (verbatim, sent as the system role) " + bannerRule
	userBanner   = bannerRule + " USER MESSAGE (verbatim, sent as the user role) " + bannerRule
	// ResponseBanner introduces what the model sent back.
	ResponseBanner = bannerRule + " MODEL RESPONSE " + bannerRule
)

// String is the whole prompt as one blob, for debugging and for the
// MACONDO_NO_LLM path. Sections run in the order the request carries them:
// tools, then the system message, then the user message.
func (p *Prompt) String() string {
	var ss strings.Builder
	if len(p.Tools) > 0 {
		ss.WriteString(toolsBanner + "\n" + ToolSchemas(p.Tools))
	}
	ss.WriteString(systemBanner + "\n" + p.System + "\n")
	ss.WriteString(userBanner + "\n" + p.User)
	return ss.String()
}

// ToolSchemas renders the tool definitions as JSON Schema, the way the
// OpenAI-compatible providers serialize them. Gemini builds an equivalent
// genai.Schema out of the same three pieces - name, description, parameters -
// so this is the shape of what every provider sends, not the exact bytes of
// any one of them.
func ToolSchemas(tools []interfaces.Tool) string {
	defs := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		properties := map[string]any{}
		required := []string{}
		for name, param := range t.Parameters() {
			spec := map[string]any{"type": param.Type, "description": param.Description}
			if param.Default != nil {
				spec["default"] = param.Default
			}
			if param.Enum != nil {
				spec["enum"] = param.Enum
			}
			if param.Items != nil {
				spec["items"] = map[string]any{"type": param.Items.Type}
			}
			properties[name] = spec
			if param.Required {
				required = append(required, name)
			}
		}
		// Parameters() is a map, so this order is otherwise whatever Go feels
		// like today. Sorting keeps two dumps of the same position diffable.
		sort.Strings(required)
		defs = append(defs, map[string]any{
			"name":        t.Name(),
			"description": t.Description(),
			"parameters": map[string]any{
				"type": "object", "properties": properties, "required": required,
			},
		})
	}
	b, err := json.MarshalIndent(defs, "", "  ")
	if err != nil {
		return "(could not render tool schemas: " + err.Error() + ")\n"
	}
	return string(b) + "\n"
}

// Notes is Macondo's own commentary to print above a prompt dump: what the
// prompt was for, and which concept cards the position pulled in. It ends with
// a banner, so everything it says is bounded by position rather than by any
// per-line marker. headline, if given, says what the prompt was about.
func (p *Prompt) Notes(headline string) string {
	var ss strings.Builder
	ss.WriteString(notesBanner + "\n")
	ss.WriteString("Each " + bannerRule + " banner labels everything below it, up to the next banner.\n")
	if headline != "" {
		ss.WriteString(headline + "\n")
	}
	ss.WriteString("Concept cards selected for this position: " +
		strings.Join(p.Concepts, ", ") + "\n")
	return ss.String()
}

// BuildPrompt assembles the prompt for a position. Which concept cards it
// includes is decided by the facts, not by the model and not by a human
// guessing which ones might come up.
func BuildPrompt(f *PositionFacts, quirky bool) (*Prompt, error) {
	cards, err := SelectConcepts(f.Flags)
	if err != nil {
		return nil, err
	}

	// The system half is deliberately free of anything position-specific, so
	// it stays identical between calls.
	quirkyText := ""
	if quirky {
		quirkyText = quirkyMD
	}
	system := strings.ReplaceAll(systemPromptMD, "{quirky}", quirkyText)

	var user strings.Builder
	if len(cards) > 0 {
		user.WriteString("## Concepts that apply to this position\n\n")
		for _, c := range cards {
			user.WriteString(c.Body)
			user.WriteString("\n\n")
		}
	}
	user.WriteString("## The position\n\n")
	user.WriteString(f.Render())
	user.WriteString("\n## " + f.Ask() + "\n")

	return &Prompt{
		System:   system,
		User:     user.String(),
		Concepts: ConceptIDs(cards),
	}, nil
}
