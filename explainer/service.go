package explainer

import (
	"context"
	"fmt"
	"os"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
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
	Provider         string // "gemini" or "openai"
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

	// Add the tools instruction
	toolsPrompt := `
	## Tools

	You have several tools at your disposal. You *must* call at least one of these tools before your explanation is complete.

	- get_our_play_metadata(play_string) -- You can call it like get_play_metadata("5D (S)PIC(A)"). This will give you data about this play such as the number of tiles it uses, the score, the vowel/consonant balance of the leave, whether it is a bingo or not. Call this tool if you wish to talk about any of these aspects of a play; do not guess or count tiles yourself!
	- get_our_future_play_metadata(play_string). Only call this function for _our own_ future or follow-up play. This function will tell you what tile draws are necessary for us to make this follow-up play, as well as whether the follow-up play needs a specific opponent play to be made first, or whether it requires the best play to be made first. It will also tell you if it's a bingo, what its score is, etc. Call it whenever you wish to talk about our _next_ play.
	Note that if this function tells you that an opponent play is required for us to make our next play, you should mention this in your explanation, when you talk about possible plays that we can make, if you choose to talk about this play. This is important context for the user to understand how this play may be possible (it could turn out that the play would be illegal to make otherwise, and we don't want to confuse the user). If it tells you that it requires the best play to be made first, then talk about this too, you can frame it as a setup opportunity, as this next play would not be possible without us making the best play first.
	- evaluate_leave(leave) -- Evaluates the value of a leave (tiles remaining on rack after a play). Takes a string of tiles like "AEINRT" and returns a numerical value. A leave should not be thought of as good until it's at least worth +2 to +3. A really strong leave can be +8 or above. Negative values indicate poor leaves. Call this tool when you want to discuss the quality of tiles remaining after a play.

	Note that you can call these tools multiple times if you wish to analyze multiple plays. You can also call them in any order. You must call at least one of these tools before your explanation is complete.
	`
	fullPrompt := prompt + "\n" + toolsPrompt

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
	default:
		return nil, fmt.Errorf("unsupported provider: %s", s.config.Provider)
	}
	log.Debug().Msg("Full Prompt:\n" + fullPrompt) // DEBUG
	response := ""
	// Generate with tools
	if os.Getenv("MACONDO_NO_LLM") == "1" {
		response = fullPrompt
	} else {
		response, err = client.GenerateWithTools(ctx, fullPrompt, s.tools,
			interfaces.WithMaxIterations(5))
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
