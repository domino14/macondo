package explainer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/Ingenimax/agent-sdk-go/pkg/llm/openai"
	"github.com/Ingenimax/agent-sdk-go/pkg/logging"
	oai "github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"
	"github.com/rs/zerolog/log"
)

// createLlamaCppClient talks to llama-server through its OpenAI-compatible API.
// It is a provider of its own for the same reason OpenRouter
// is: pointing openai-base-url at a local server would hand it the real OpenAI
// key, and would leave later OpenAI explanations going to localhost.
//
// The SDK's clients are built without any way to pass request options, so the
// chat service is replaced outright. It is the only service the tool-calling
// path uses.
func (s *Service) createLlamaCppClient() (interfaces.LLM, error) {
	log.Info().Str("model", s.config.Model).Str("base_url", s.config.BaseURL).Msg("Using llama.cpp model")
	client := openai.NewClient(
		s.config.APIKey,
		openai.WithModel(s.config.Model),
		openai.WithBaseURL(s.config.BaseURL),
		openai.WithLogger(logging.New()),
	)
	client.ChatService = oai.NewChatService(
		option.WithAPIKey(s.config.APIKey),
		option.WithBaseURL(s.config.BaseURL),
		option.WithMiddleware(systemMessagesFirst),
	)
	return client, nil
}

// afterThinking returns what follows the last </think> in a reply. A server only
// separates a model's reasoning from its answer when the reply has the shape it
// expects, and with thinking off Qwen sometimes writes a draft, a stray </think>
// and then the real answer, all as content. A reply with nothing after the tag
// comes back whole, since showing the draft beats showing nothing.
func afterThinking(reply string) string {
	const closeTag = "</think>"
	i := strings.LastIndex(reply, closeTag)
	if i < 0 {
		return reply
	}
	if answer := strings.TrimSpace(reply[i+len(closeTag):]); answer != "" {
		return answer
	}
	return reply
}

// systemMessagesFirst rewrites a chat request so that its system message comes
// first. The SDK appends the system message after the user's, and its last-ditch
// call at the iteration limit appends another at the very end. Hosted APIs
// accept that, but local servers render messages through the model's own chat
// template, and some of those refuse: Qwen's raises "System message must be at
// the beginning", which llama-server reports as a 500.
func systemMessagesFirst(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	if req.Body == nil {
		return next(req)
	}
	body, err := io.ReadAll(req.Body)
	req.Body.Close()
	if err != nil {
		return nil, fmt.Errorf("reading chat request: %w", err)
	}
	body, err = hoistSystemMessages(body)
	if err != nil {
		return nil, fmt.Errorf("reordering chat request: %w", err)
	}
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(body)), nil
	}
	return next(req)
}

// hoistSystemMessages merges every system message in a chat request into one at
// the front, keeping the rest of the conversation in order. Merging rather than
// moving because the templates that insist on a leading system message allow
// only that one. A request that already has this shape comes back untouched.
func hoistSystemMessages(body []byte) ([]byte, error) {
	var request map[string]json.RawMessage
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, err
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(request["messages"], &messages); err != nil {
		return nil, err
	}

	var system []string
	rest := make([]json.RawMessage, 0, len(messages))
	inPlace := true
	for i, raw := range messages {
		var m struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, err
		}
		if m.Role != "system" {
			rest = append(rest, raw)
			continue
		}
		var text string
		if err := json.Unmarshal(m.Content, &text); err != nil {
			return nil, fmt.Errorf("system message content is not text: %w", err)
		}
		system = append(system, text)
		inPlace = inPlace && i == 0
	}
	if len(system) == 0 || (len(system) == 1 && inPlace) {
		return body, nil
	}

	merged, err := json.Marshal(map[string]string{
		"role":    "system",
		"content": strings.Join(system, "\n\n"),
	})
	if err != nil {
		return nil, err
	}
	reordered, err := json.Marshal(append([]json.RawMessage{merged}, rest...))
	if err != nil {
		return nil, err
	}
	request["messages"] = reordered
	return json.Marshal(request)
}
