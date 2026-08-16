package explainer

import (
	"context"
	"fmt"
	"os"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/Ingenimax/agent-sdk-go/pkg/llm/deepseek"
	"github.com/Ingenimax/agent-sdk-go/pkg/llm/gemini"
	"github.com/Ingenimax/agent-sdk-go/pkg/llm/openai"
	"github.com/Ingenimax/agent-sdk-go/pkg/logging"
	"github.com/domino14/macondo/ai/bot"
	macondo "github.com/domino14/macondo/config"
	"github.com/domino14/macondo/montecarlo"
	"github.com/domino14/macondo/montecarlo/stats"
	"github.com/rs/zerolog/log"
	"google.golang.org/genai"
)

// Config holds configuration for the explainer service
type Config struct {
	Provider  string // "gemini", "openai", or "deepseek"
	APIKey    string
	Model     string
	BaseURL   string // optional: override base URL (e.g. for LM Studio at http://127.0.0.1:1234/v1)
	UseQuirky bool
}

// Service provides the main explainer service
type Service struct {
	config   *Config
	analyzer *Analyzer
	tools    []interfaces.Tool
	last     *Exchange
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

	config := DefaultConfig(macondoConfig)

	return &Service{
		config:   config,
		analyzer: analyzer,
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
}

// ExplainResult contains the explanation from the AI
type ExplainResult struct {
	Explanation  string
	InputTokens  int
	OutputTokens int
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

	facts, err := s.analyzer.BuildFacts(in.Simmer, in.SimStats, in.Compare)
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

	result := &ExplainResult{Concepts: prompt.Concepts, Prompt: prompt}
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
		resp, err := client.GenerateWithToolsDetailed(ctx, prompt.User, s.tools,
			interfaces.WithSystemMessage(prompt.System),
			interfaces.WithMaxIterations(7))
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

func comparisonLabel(f *PositionFacts) string {
	if f.Comparison == nil || f.Comparison.Rival == nil {
		return ""
	}
	if f.Comparison.WasBest {
		return f.Comparison.Rival.Play + " (the runner-up; they found the best play)"
	}
	return f.Comparison.Rival.Play
}

func (s *Service) createClient(ctx context.Context) (interfaces.LLM, error) {
	switch s.config.Provider {
	case "gemini":
		return s.createGeminiClient(ctx)
	case "openai":
		log.Info().Msg("Using OpenAI client")
		return s.createOpenAIClient()
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
