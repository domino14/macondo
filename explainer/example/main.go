package main

import (
	"context"
	"fmt"
	"log"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/explainer"
)

func main() {
	ctx := context.Background()

	// Example game situation data (would normally come from macondo simulation)
	gameState := `Game state:
Player 1: 123 points
Player 2: 145 points
Current turn: Player 1
Tiles on rack: AEINRST
Tiles in bag: 42`

	simResults := `Play         Equity    Win%      Score    Tiles
8D RETAINS   45.2±2.1  52.3%±1.5  68      AEINRST
8H ANTSIER   42.1±1.8  48.7%±1.2  72      AEINRST
7F RATINES   41.5±2.0  47.2%±1.4  65      AEINRST
6G STAINER   40.8±1.9  46.5%±1.3  70      AEINRST
5E RETINAS   39.2±1.7  45.1%±1.1  64      AEINRST`

	simDetails := `**Ply 1 - Opponent's response
Play         Score    Probability
---------------------------------
12A QUIET    42       8.5%
11B ZAX      38       6.2%
10C JOKE     35       5.8%
9D VEND      32       4.9%
8E FLOW      28       4.1%

**Ply 2 - Our follow-up
Play         Score    Probability    Needed Draw
-------------------------------------------------
15A BINGOS   86       12.3%          {GO}
14B TRAINS   78       9.8%           {none}
13C SATIRE   72       8.5%           {none}
12D RETINA   68       7.2%           {none}
11E INERTS   65       6.8%           {none}`

	winningPlay := "8D RETAINS"

	winningStats := `### Opponent's next play
Play         Score    Probability
---------------------------------
12A QUIET    42       8.5%
11B ZAX      38       6.2%
10C JOKE     35       5.8%

### Our follow-up play
Play         Score    Probability    Needed Draw
-------------------------------------------------
15A BINGOS   86       12.3%          {GO}
14B TRAINS   78       9.8%           {none}
13C SATIRE   72       8.5%           {none}`

	// Load macondo configuration
	macondoConfig := config.DefaultConfig()
	err := macondoConfig.Load(nil)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Create the service
	service := explainer.NewService(macondoConfig)

	// Generate explanation
	provider := macondoConfig.GetString("genai-provider")
	var model string
	if provider == "openai" {
		model = macondoConfig.GetString("openai-model")
	} else {
		model = macondoConfig.GetString("gemini-model")
	}
	fmt.Printf("Generating explanation using %s provider with model %s...\n", provider, model)

	result, err := service.Explain(
		ctx,
		gameState,
		simResults,
		simDetails,
		winningPlay,
		winningStats,
	)
	if err != nil {
		log.Fatalf("Failed to generate explanation: %v", err)
	}

	// Display the result
	fmt.Println("\n=== Explanation ===")
	fmt.Println(result.Explanation)

	if result.InputTokens > 0 {
		fmt.Printf("\nInput tokens: %d\n", result.InputTokens)
		fmt.Printf("Output tokens: %d\n", result.OutputTokens)
	}
}
