package explainer

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/Ingenimax/agent-sdk-go/pkg/llm/deepseek"
	"github.com/Ingenimax/agent-sdk-go/pkg/llm/gemini"
	"github.com/Ingenimax/agent-sdk-go/pkg/llm/openai"
	"github.com/Ingenimax/agent-sdk-go/pkg/logging"
	"github.com/domino14/macondo/ai/bot"
	macondo "github.com/domino14/macondo/config"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/montecarlo/stats"
	oai "github.com/openai/openai-go/v2"
	"github.com/rs/zerolog/log"
	"google.golang.org/genai"
)

// Config holds configuration for the explainer service
type Config struct {
	Provider  string // "gemini", "openai", "openrouter", or "deepseek"
	APIKey    string
	Model     string
	BaseURL   string // optional: override base URL (e.g. for LM Studio at http://127.0.0.1:1234/v1)
	UseQuirky bool
}

// OpenRouter is an OpenAI-compatible endpoint sitting in front of every other
// vendor's models, so it needs no client of its own - only a base URL. It gets
// a *provider* of its own anyway, because the alternative is telling people to
// put an OpenRouter key in openai-api-key, which overwrites the OpenAI key they
// may already be using. The two are different accounts and want different slots.
const (
	openRouterBaseURL = "https://openrouter.ai/api/v1"
	// A free model, so that a new key explains a position without any further
	// configuration. Free models come and go; when this one goes, OpenRouter
	// says so plainly and `setconfig openrouter-model` picks another.
	openRouterDefaultModel = "google/gemma-4-31b-it:free"
)

// Service provides the main explainer service
type Service struct {
	// macondoConfig is kept so that every explanation re-reads the provider
	// settings. The shell builds one Service and reuses it, so without this a
	// `setconfig genai-provider ...` did nothing until the shell was restarted -
	// which is most of the friction in trying a few models to see which
	// explains best.
	macondoConfig *macondo.Config
	config        *Config
	analyzer      *Analyzer
	tools         []interfaces.Tool
	last          *Exchange
}

// Exchange is what was last sent to the model and what came back. The prompt
// is assembled from so many computed pieces that being able to read the exact
// thing that produced an explanation is the only practical way to debug one.
type Exchange struct {
	Prompt   *Prompt
	Response string
	// BestPlay and Comparison say what the explanation was about, so a dump
	// after the fact is identifiable.
	BestPlay   string
	Comparison string
}

// LastExchange returns the most recent prompt and response, or nil if this
// service hasn't explained anything yet.
func (s *Service) LastExchange() *Exchange {
	return s.last
}

// String renders the exchange for a human to read: our notes, then the two
// messages exactly as sent, then the reply. Each section is introduced by a
// banner and runs to the next one.
func (e *Exchange) String() string {
	headline := "Explanation of " + e.BestPlay
	if e.Comparison != "" {
		headline += ", compared against " + e.Comparison
	}
	return e.Prompt.Notes(headline) + e.Prompt.String() +
		"\n" + ResponseBanner + "\n" + e.Response + "\n"
}

// NewService creates a new explainer service
func NewService(macondoConfig *macondo.Config) *Service {
	analyzer := NewAnalyzer()
	analyzer.SetConfig(macondoConfig)

	return &Service{
		macondoConfig: macondoConfig,
		config:        DefaultConfig(macondoConfig),
		analyzer:      analyzer,
		tools: []interfaces.Tool{
			NewGetOurPlayMetadataTool(analyzer),
			NewGetOurFuturePlayMetadataTool(analyzer),
			NewEvaluateLeaveTool(analyzer),
		},
	}
}

// ExplainInput is the position to explain. The service builds everything it
// sends to the model from these three, so nobody has to pre-render tables and
// nobody has to parse them back.
type ExplainInput struct {
	Game     *bot.BotTurnPlayer
	Simmer   *montecarlo.Simmer
	SimStats *stats.SimStats
	// Compare asks for a head-to-head against a particular play, usually the
	// one the player actually made. The simulation must already have
	// evaluated it - see Simmer.AvoidPruningMoves.
	Compare *ComparisonRequest
	// Inference is the read on the opponent's rack that Simmer was run with,
	// plus the same plays simmed without it. Nil when no read was taken.
	Inference *InferenceInput
	// Model overrides the configured model for this one explanation, so that
	// several can be tried on the same position without editing any config.
	// Empty means whatever the provider is configured to use.
	Model string
}

// ExplainResult contains the explanation from the AI
type ExplainResult struct {
	Explanation  string
	InputTokens  int
	OutputTokens int
	// Provider and Model say who actually answered. Worth reporting because
	// both can now change between one explanation and the next.
	Provider string
	Model    string
	// Concepts names the concept cards this position pulled into the prompt.
	Concepts []string
	// Prompt is exactly what was sent. Also kept on the service, so it can be
	// read back after the fact.
	Prompt *Prompt
}

// Explain generates an explanation for a finished simulation.
func (s *Service) Explain(ctx context.Context, in *ExplainInput) (*ExplainResult, error) {
	if in == nil || in.Game == nil || in.Simmer == nil || in.SimStats == nil {
		return nil, fmt.Errorf("explain needs a game, a simmer and sim stats")
	}
	s.analyzer.SetGame(in.Game)
	s.refreshConfig(in.Model)

	facts, err := s.analyzer.BuildFacts(in.Simmer, in.SimStats, in.Compare, in.Inference)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze position: %w", err)
	}

	prompt, err := BuildPrompt(facts, s.config.UseQuirky)
	if err != nil {
		return nil, fmt.Errorf("failed to build prompt: %w", err)
	}
	// The tools go out with every request, so they belong in any dump of what
	// was sent, even though they travel outside the messages.
	prompt.Tools = s.tools
	log.Debug().Strs("concepts", prompt.Concepts).Msg("selected concept cards")
	log.Debug().Msg("Full prompt:\n" + prompt.String())

	result := &ExplainResult{
		Concepts: prompt.Concepts, Prompt: prompt,
		Provider: s.config.Provider, Model: s.config.Model,
	}
	response := ""
	if os.Getenv("MACONDO_NO_LLM") == "1" {
		// Printing the prompt is the whole point of this mode, so it becomes
		// the "explanation". What gets kept as the response says what
		// happened instead, or reading it back later would show the prompt
		// twice.
		result.Explanation = prompt.String()
		response = "(MACONDO_NO_LLM=1 was set: no model was called. " +
			"What the command printed was the prompt itself.)"
	} else {
		client, err := s.createClient(ctx)
		if err != nil {
			return nil, err
		}
		resp, err := generateWithRetry(ctx, client, prompt, s.tools)
		if err != nil {
			return nil, fmt.Errorf("failed to generate explanation: %w", err)
		}
		result.Explanation = resp.Content
		response = resp.Content
		if resp.Usage != nil {
			result.InputTokens = resp.Usage.InputTokens
			result.OutputTokens = resp.Usage.OutputTokens
		}
	}

	s.last = &Exchange{
		Prompt:     prompt,
		Response:   response,
		BestPlay:   facts.Best.Play,
		Comparison: comparisonLabel(facts),
	}
	return result, nil
}

const (
	// maxToolIterations bounds the tool-calling loop: each round trip is a
	// whole request, so this is also the worst case for what one explanation
	// costs, both in money and against a per-day request quota.
	maxToolIterations = 7
	// explainAttempts counts the first try, so 3 means two retries.
	explainAttempts = 3
	// explainRetryCap is the longest Retry-After worth honouring. A provider
	// asking for more than this is not briefly busy - it is shut to us for now,
	// and someone waiting at a prompt would rather hear that than sit through
	// it.
	explainRetryCap = 30 * time.Second
)

// explainRetryWait is the pause before the second attempt, doubling after
// that. A provider's own Retry-After takes precedence when it sends one. It is
// a var so that its own tests don't have to spend it.
var explainRetryWait = 2 * time.Second

// transientStatuses are the HTTP statuses worth sending again. A shared free
// pool hands out 429s constantly - the free endpoints have one upstream each
// and no failover - and OpenRouter forwards a provider's own 5xx as its own.
// Everything else (a rejected key, an unknown model, a malformed request)
// fails identically however many times it is sent, and retrying one only burns
// requests from a daily quota.
var transientStatuses = map[int]bool{
	http.StatusTooManyRequests:     true,
	http.StatusInternalServerError: true,
	http.StatusBadGateway:          true,
	http.StatusServiceUnavailable:  true,
	http.StatusGatewayTimeout:      true,
}

// generateWithRetry asks the model for an explanation, trying again when the
// failure is the kind that passes. The SDK has a retry policy of its own, but
// it is only wired into Generate and Chat - never into the tool-calling path
// this uses - so retrying is ours to do.
//
// Only the OpenAI-protocol providers (openai, openrouter, and any local server
// behind them) report a status code we can read. Gemini and DeepSeek errors
// come back untyped and so are never retried.
func generateWithRetry(ctx context.Context, client interfaces.LLM, prompt *Prompt,
	tools []interfaces.Tool) (*interfaces.LLMResponse, error) {

	backoff := explainRetryWait
	for attempt := 1; ; attempt++ {
		resp, err := client.GenerateWithToolsDetailed(ctx, prompt.User, tools,
			interfaces.WithSystemMessage(prompt.System),
			interfaces.WithMaxIterations(maxToolIterations))
		if err == nil {
			return resp, nil
		}

		asked, retryable := retryableFailure(err)
		if !retryable || attempt == explainAttempts {
			return nil, err
		}
		wait := backoff
		if asked > 0 {
			wait = asked
		}
		log.Warn().Err(err).Int("attempt", attempt).Int("of", explainAttempts).
			Dur("retrying_in", wait).Msg("explanation request failed; trying again")

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(wait):
		}
		backoff *= 2
	}
}

// retryableFailure reports whether err is worth another attempt, and how long
// the provider asked us to wait before it. A zero wait means it didn't say and
// the caller should use its own backoff.
func retryableFailure(err error) (wait time.Duration, retryable bool) {
	var apiErr *oai.Error
	if !errors.As(err, &apiErr) || !transientStatuses[apiErr.StatusCode] {
		return 0, false
	}
	if apiErr.Response == nil {
		return 0, true
	}
	secs, convErr := strconv.Atoi(apiErr.Response.Header.Get("Retry-After"))
	if convErr != nil || secs < 0 {
		return 0, true
	}
	if asked := time.Duration(secs) * time.Second; asked <= explainRetryCap {
		return asked, true
	}
	return 0, false
}

func comparisonLabel(f *PositionFacts) string {
	if f.Comparison == nil || f.Comparison.Rival == nil {
		return ""
	}
	if f.Comparison.WasBest {
		return f.Comparison.Rival.Play + " (the runner-up; they found the best play)"
	}
	return f.Comparison.Rival.Play
}

// refreshConfig re-reads the provider settings, so that a `setconfig` takes
// effect on the next explanation rather than on the next shell. model, when
// given, replaces the configured one for this call alone - the re-read is what
// makes that temporary, since the next call derives everything again.
func (s *Service) refreshConfig(model string) {
	if s.macondoConfig != nil {
		s.config = DefaultConfig(s.macondoConfig)
	}
	if model != "" {
		s.config.Model = model
	}
}

func (s *Service) createClient(ctx context.Context) (interfaces.LLM, error) {
	switch s.config.Provider {
	case "gemini":
		return s.createGeminiClient(ctx)
	case "openai":
		log.Info().Msg("Using OpenAI client")
		return s.createOpenAIClient()
	case "openrouter":
		log.Info().Msg("Using OpenRouter client")
		return s.createOpenRouterClient()
	case "deepseek":
		log.Info().Msg("Using DeepSeek client")
		return s.createDeepSeekClient()
	default:
		return nil, fmt.Errorf("unsupported provider: %s", s.config.Provider)
	}
}

func (s *Service) createGeminiClient(ctx context.Context) (interfaces.LLM, error) {
	authOption := gemini.WithAPIKey(s.config.APIKey)
	backendOption := gemini.WithBackend(genai.BackendGeminiAPI)

	model := s.config.Model
	if model == "" {
		model = "gemini-2.5-flash"
	}
	log.Info().Str("model", model).Msg("Using Gemini model")
	return gemini.NewClient(ctx, authOption, backendOption, gemini.WithModel(model))
}

func (s *Service) createOpenAIClient() (interfaces.LLM, error) {
	model := s.config.Model
	logger := logging.New()

	if model == "" {
		model = "gpt-4.1"
	}
	opts := []openai.Option{
		openai.WithModel(model),
		openai.WithLogger(logger),
	}
	if s.config.BaseURL != "" {
		log.Info().Str("base_url", s.config.BaseURL).Msg("Using custom OpenAI base URL")
		opts = append(opts, openai.WithBaseURL(s.config.BaseURL))
	}
	log.Info().Str("model", model).Msg("Using OpenAI model")
	return openai.NewClient(s.config.APIKey, opts...), nil
}

func (s *Service) createOpenRouterClient() (interfaces.LLM, error) {
	model := s.config.Model
	if model == "" {
		model = openRouterDefaultModel
	}
	baseURL := s.config.BaseURL
	if baseURL == "" {
		baseURL = openRouterBaseURL
	}
	// OpenRouter names a model "vendor/model", with an optional ":free" or
	// other variant on the end. A bare "gpt-4.1" is an OpenAI name, and what
	// comes back for one is a 400 that doesn't say which part was wrong.
	if !strings.Contains(model, "/") {
		log.Warn().Str("model", model).Msg(
			"openrouter model names look like vendor/model (google/gemma-4-31b-it:free); " +
				"this one is likely to be rejected")
	}
	log.Info().Str("model", model).Str("base_url", baseURL).Msg("Using OpenRouter model")
	return openai.NewClient(
		s.config.APIKey,
		openai.WithModel(model),
		openai.WithBaseURL(baseURL),
		openai.WithLogger(logging.New()),
	), nil
}

func (s *Service) createDeepSeekClient() (interfaces.LLM, error) {
	model := s.config.Model
	logger := logging.New()

	if model == "" {
		model = "deepseek-chat"
	}
	modelOption := deepseek.WithModel(model)
	log.Info().Str("model", model).Msg("Using DeepSeek model")
	return deepseek.NewClient(
		s.config.APIKey,
		modelOption,
		deepseek.WithLogger(logger),
	), nil
}

// DefaultConfig returns a default configuration from macondo config
func DefaultConfig(macondoConfig *macondo.Config) *Config {
	provider := macondoConfig.GetString(macondo.ConfigGenaiProvider)

	var apiKey, model, baseURL string
	switch provider {
	case "openai":
		apiKey = macondoConfig.GetString(macondo.ConfigOpenaiApiKey)
		model = macondoConfig.GetString(macondo.ConfigOpenaiModel)
		baseURL = macondoConfig.GetString(macondo.ConfigOpenaiBaseURL)
	case "openrouter":
		apiKey = macondoConfig.GetString(macondo.ConfigOpenrouterApiKey)
		model = macondoConfig.GetString(macondo.ConfigOpenrouterModel)
		baseURL = macondoConfig.GetString(macondo.ConfigOpenrouterBaseURL)
	case "gemini":
		apiKey = macondoConfig.GetString(macondo.ConfigGeminiApiKey)
		model = macondoConfig.GetString(macondo.ConfigGeminiModel)
	case "deepseek":
		apiKey = macondoConfig.GetString(macondo.ConfigDeepseekApiKey)
		model = macondoConfig.GetString(macondo.ConfigDeepseekModel)
	}

	useQuirky := os.Getenv("GENAI_QUIRKY") != ""

	return &Config{
		Provider:  provider,
		APIKey:    apiKey,
		Model:     model,
		BaseURL:   baseURL,
		UseQuirky: useQuirky,
	}
}
