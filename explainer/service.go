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
	"github.com/rs/zerolog/log"
	"google.golang.org/genai"
)

// Config holds configuration for the explainer service
type Config struct {
	Provider         string // "gemini", "openai", or "deepseek"
	APIKey           string
	Model            string
	UseQuirky        bool
	MainPromptPath   string
	QuirkyPromptPath string
}

// Service provides the main explainer service
type Service struct {
	config   *Config
	analyzer *Analyzer
	tools    []interfaces.Tool
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

func (s *Service) SetGame(tp *bot.BotTurnPlayer) {
	s.analyzer.game = tp
}

// ExplainResult contains the explanation from the AI
type ExplainResult struct {
	Explanation  string
	InputTokens  int
	OutputTokens int
}

// Explain generates an explanation for the given game situation
func (s *Service) Explain(ctx context.Context, gameState, simResults, simDetails, winningPlay, winningStats string) (*ExplainResult, error) {
	// Set the game context for the analyzer
	s.analyzer.SetGameContext(gameState, simResults, simDetails, winningPlay, winningStats)

	// Build the prompt
	quirkyPath := ""
	if s.config.UseQuirky {
		quirkyPath = s.config.QuirkyPromptPath
	}

	prompt, err := s.analyzer.BuildPrompt(s.config.MainPromptPath, quirkyPath)
	if err != nil {
		return nil, fmt.Errorf("failed to build prompt: %w", err)
	}

	// Create the LLM client based on provider
	var client interfaces.LLM

	switch s.config.Provider {
	case "gemini":
		geminiClient, err := s.createGeminiClient(ctx)
		if err != nil {
			return nil, err
		}
		client = geminiClient
	case "openai":
		openaiClient, err := s.createOpenAIClient(ctx)
		if err != nil {
			return nil, err
		}
		client = openaiClient
		log.Info().Msg("Using OpenAI client")
	case "deepseek":
		deepseekClient, err := s.createDeepSeekClient(ctx)
		if err != nil {
			return nil, err
		}
		client = deepseekClient
		log.Info().Msg("Using DeepSeek client")
	default:
		return nil, fmt.Errorf("unsupported provider: %s", s.config.Provider)
	}
	log.Debug().Msg("Full Prompt:\n" + prompt) // DEBUG
	response := ""
	// Generate with tools
	if os.Getenv("MACONDO_NO_LLM") == "1" {
		response = prompt
	} else {
		response, err = client.GenerateWithTools(ctx, prompt, s.tools,
			interfaces.WithMaxIterations(7))
		// response, err := client.Generate(ctx, fullPrompt)

		if err != nil {
			return nil, fmt.Errorf("failed to generate explanation: %w", err)
		}
	}
	return &ExplainResult{
		Explanation:  response,
		InputTokens:  0, // TODO: Extract from response metadata
		OutputTokens: 0, // TODO: Extract from response metadata
	}, nil

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

func (s *Service) createOpenAIClient(ctx context.Context) (interfaces.LLM, error) {
	model := s.config.Model
	logger := logging.New()

	if model == "" {
		model = "gpt-4.1"
	}
	modelOption := openai.WithModel(model)
	log.Info().Str("model", model).Msg("Using OpenAI model")
	return openai.NewClient(
		s.config.APIKey,
		modelOption,
		openai.WithLogger(logger),
	), nil
}

func (s *Service) createDeepSeekClient(ctx context.Context) (interfaces.LLM, error) {
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

	var apiKey, model string
	switch provider {
	case "openai":
		apiKey = macondoConfig.GetString(macondo.ConfigOpenaiApiKey)
		model = macondoConfig.GetString(macondo.ConfigOpenaiModel)
	case "gemini":
		apiKey = macondoConfig.GetString(macondo.ConfigGeminiApiKey)
		model = macondoConfig.GetString(macondo.ConfigGeminiModel)
	case "deepseek":
		apiKey = macondoConfig.GetString(macondo.ConfigDeepseekApiKey)
		model = macondoConfig.GetString(macondo.ConfigDeepseekModel)
	}

	useQuirky := os.Getenv("GENAI_QUIRKY") != ""

	return &Config{
		Provider:         provider,
		APIKey:           apiKey,
		Model:            model,
		UseQuirky:        useQuirky,
		MainPromptPath:   "explainer/main_prompt.md",
		QuirkyPromptPath: "explainer/quirky.md",
	}
}
