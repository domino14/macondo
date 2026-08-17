package explainer

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/matryer/is"
	oai "github.com/openai/openai-go/v2"
)

// quickBackoff keeps the waiting out of the test suite. What is being tested
// is which failures come back and how often, not how patient the pauses are.
func quickBackoff(t *testing.T) {
	t.Helper()
	was := explainRetryWait
	explainRetryWait = time.Millisecond
	t.Cleanup(func() { explainRetryWait = was })
}

// apiError builds the error an OpenAI-protocol provider returns, complete
// enough to be logged: Error() reads both the request and the response.
func apiError(status int, retryAfter string) error {
	h := http.Header{}
	if retryAfter != "" {
		h.Set("Retry-After", retryAfter)
	}
	return &oai.Error{
		StatusCode: status,
		Request:    httptest.NewRequest("POST", "https://openrouter.ai/api/v1/chat/completions", nil),
		Response:   &http.Response{StatusCode: status, Header: h},
	}
}

// fakeLLM answers with whatever the script says, one entry per attempt. Only
// the tool-calling call is implemented; embedding the interface covers the
// rest, and a test that reached one would panic rather than pass quietly.
type fakeLLM struct {
	interfaces.LLM
	errs  []error // errs[i] is what attempt i+1 returns
	calls int
}

func (f *fakeLLM) GenerateWithToolsDetailed(ctx context.Context, prompt string,
	tools []interfaces.Tool, opts ...interfaces.GenerateOption) (*interfaces.LLMResponse, error) {

	f.calls++
	if f.calls <= len(f.errs) {
		if err := f.errs[f.calls-1]; err != nil {
			return nil, err
		}
	}
	return &interfaces.LLMResponse{Content: "12K QU(ID) is best because..."}, nil
}

// A 429 from a shared free pool is the normal weather on a free endpoint, not
// a reason to throw away a finished simulation.
func TestARateLimitIsTriedAgain(t *testing.T) {
	is := is.New(t)
	quickBackoff(t)

	llm := &fakeLLM{errs: []error{apiError(http.StatusTooManyRequests, "")}}
	resp, err := generateWithRetry(context.Background(), llm, &Prompt{User: "u", System: "s"}, nil)
	is.NoErr(err)
	is.Equal(llm.calls, 2)
	is.True(resp.Content != "")
}

// Two retries and then the error is the user's problem, rather than a command
// that sits there resending forever.
func TestRetriesGiveUp(t *testing.T) {
	is := is.New(t)
	quickBackoff(t)

	boom := apiError(http.StatusServiceUnavailable, "")
	llm := &fakeLLM{errs: []error{boom, boom, boom, boom}}
	_, err := generateWithRetry(context.Background(), llm, &Prompt{}, nil)
	is.True(errors.Is(err, boom))
	is.Equal(llm.calls, explainAttempts)
}

// A rejected key or an unknown model fails the same way however many times it
// is sent, and each attempt spends a request from a 50-a-day quota.
func TestPermanentFailuresAreNotRetried(t *testing.T) {
	is := is.New(t)

	for _, status := range []int{
		http.StatusBadRequest, http.StatusUnauthorized,
		http.StatusForbidden, http.StatusNotFound,
	} {
		llm := &fakeLLM{errs: []error{apiError(status, "")}}
		_, err := generateWithRetry(context.Background(), llm, &Prompt{}, nil)
		is.True(err != nil)
		is.Equal(llm.calls, 1)
	}

	// Gemini and DeepSeek report failures untyped, so nothing can be read from
	// them and nothing is resent.
	llm := &fakeLLM{errs: []error{errors.New("quota exceeded")}}
	_, err := generateWithRetry(context.Background(), llm, &Prompt{}, nil)
	is.True(err != nil)
	is.Equal(llm.calls, 1)
}

func TestRetryableFailureReadsRetryAfter(t *testing.T) {
	is := is.New(t)

	// No header: retry, on our own schedule.
	wait, retry := retryableFailure(apiError(http.StatusTooManyRequests, ""))
	is.True(retry)
	is.Equal(wait, time.Duration(0))

	// A provider that says when to come back is obeyed.
	wait, retry = retryableFailure(apiError(http.StatusTooManyRequests, "5"))
	is.True(retry)
	is.Equal(wait, 5*time.Second)

	// Junk in the header is not a reason to abandon the attempt.
	wait, retry = retryableFailure(apiError(http.StatusTooManyRequests, "in a bit"))
	is.True(retry)
	is.Equal(wait, time.Duration(0))

	// An hour is not "briefly busy". Someone waiting at a prompt would rather
	// be told than sat through it.
	_, retry = retryableFailure(apiError(http.StatusTooManyRequests, "3600"))
	is.True(!retry)

	// The classification survives the wrapping the SDK and we both add.
	wrapped := fmt.Errorf("failed to generate explanation: %w",
		fmt.Errorf("failed to create chat completion: %w",
			apiError(http.StatusBadGateway, "")))
	_, retry = retryableFailure(wrapped)
	is.True(retry)
}

// A cancelled context stops the waiting, so ^C is not ignored for the length
// of a backoff.
func TestRetryStopsWhenTheContextIsDone(t *testing.T) {
	is := is.New(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	llm := &fakeLLM{errs: []error{apiError(http.StatusTooManyRequests, "")}}
	_, err := generateWithRetry(ctx, llm, &Prompt{}, nil)
	is.True(errors.Is(err, context.Canceled))
	is.Equal(llm.calls, 1)
}
