package explainer

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/domino14/macondo/config"
	"github.com/matryer/is"
)

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func messagesOf(t *testing.T, body []byte) []chatMessage {
	t.Helper()
	var request struct {
		Messages []chatMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatalf("bad request body %s: %v", body, err)
	}
	return request.Messages
}

func TestHoistSystemMessages(t *testing.T) {
	sys := chatMessage{"system", "rules"}
	user := chatMessage{"user", "explain"}
	assistant := chatMessage{"assistant", "thinking it over"}
	tool := chatMessage{"tool", "3.2"}

	for _, tc := range []struct {
		name string
		in   []chatMessage
		want []chatMessage
	}{
		{"the order the SDK sends", []chatMessage{user, sys}, []chatMessage{sys, user}},
		{"already first", []chatMessage{sys, user}, []chatMessage{sys, user}},
		{"no system message", []chatMessage{user, assistant}, []chatMessage{user, assistant}},
		{
			"the SDK's call at the iteration limit",
			[]chatMessage{user, sys, assistant, tool, {"system", "conclude"}},
			[]chatMessage{{"system", "rules\n\nconclude"}, user, assistant, tool},
		},
		{
			"two already at the front still become one",
			[]chatMessage{sys, {"system", "more"}, user},
			[]chatMessage{{"system", "rules\n\nmore"}, user},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			is := is.New(t)
			in, err := json.Marshal(map[string]any{"model": "m", "messages": tc.in})
			is.NoErr(err)

			out, err := hoistSystemMessages(in)
			is.NoErr(err)
			is.Equal(messagesOf(t, out), tc.want)

			var request map[string]any
			is.NoErr(json.Unmarshal(out, &request))
			is.Equal(request["model"], "m") // the rest of the request survives
		})
	}
}

func TestAfterThinking(t *testing.T) {
	for _, tc := range []struct{ name, in, want string }{
		{"no tag", "the answer", "the answer"},
		{"draft then answer", "a draft\n</think>\n\nthe answer\n", "the answer"},
		{"the last tag wins", "one</think>two</think>three", "three"},
		{"nothing after the tag", "only a draft\n</think>\n", "only a draft\n</think>\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			is.New(t).Equal(afterThinking(tc.in), tc.want)
		})
	}
}

// Fields the middleware doesn't know about have to reach the server as sent. A
// tool call is the one that matters: dropping it would break the tool loop.
func TestHoistSystemMessagesKeepsOtherFields(t *testing.T) {
	is := is.New(t)
	in := `{"tools":[{"type":"function"}],"messages":[
		{"role":"assistant","content":"","tool_calls":[{"id":"call_1"}]},
		{"role":"tool","content":"3.2","tool_call_id":"call_1"},
		{"role":"system","content":"rules"}]}`

	out, err := hoistSystemMessages([]byte(in))
	is.NoErr(err)

	var request struct {
		Tools    []map[string]any `json:"tools"`
		Messages []map[string]any `json:"messages"`
	}
	is.NoErr(json.Unmarshal(out, &request))
	is.Equal(len(request.Tools), 1)
	is.Equal(request.Messages[0]["role"], "system")
	is.Equal(request.Messages[1]["tool_calls"].([]any)[0].(map[string]any)["id"], "call_1")
	is.Equal(request.Messages[2]["tool_call_id"], "call_1")
}

// The whole path, through the real SDK client, to a server that behaves the way
// llama-server does with Qwen's chat template.
func TestLlamaCppProviderSendsSystemMessageFirst(t *testing.T) {
	is := is.New(t)

	var auth, model string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		messages := messagesOf(t, body)
		if len(messages) == 0 || messages[0].Role != "system" {
			http.Error(w, `{"error":{"code":500,"message":"System message must be at the beginning."}}`,
				http.StatusInternalServerError)
			return
		}
		var request struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(body, &request); err != nil {
			t.Error(err)
		}
		auth, model = r.Header.Get("Authorization"), request.Model
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"1","object":"chat.completion","choices":[{"index":0,
			"finish_reason":"stop","message":{"role":"assistant","content":"because"}}],
			"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8}}`)
	}))
	defer server.Close()

	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigGenaiProvider, "llamacpp")
	cfg.Set(config.ConfigLlamaCppBaseURL, server.URL+"/v1")
	cfg.Set(config.ConfigLlamaCppModel, "qwen")
	cfg.Set(config.ConfigLlamaCppApiKey, "llamacpp-key")
	cfg.Set(config.ConfigOpenaiApiKey, "real-openai-key")

	svc := NewService(cfg)
	client, err := svc.createClient(context.Background())
	is.NoErr(err)

	resp, err := generateWithRetry(context.Background(), client,
		&Prompt{System: "rules", User: "explain"}, nil)
	is.NoErr(err)
	is.Equal(resp.Content, "because")
	is.Equal(model, "qwen")
	is.Equal(auth, "Bearer llamacpp-key") // and not the OpenAI key
}
