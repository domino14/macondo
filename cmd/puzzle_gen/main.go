//go:build ignore

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/game"
	"github.com/domino14/macondo/gcgio"
	pb "github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/domino14/macondo/puzzles"
	"github.com/rs/zerolog"
)

var gameIDs = []string{
	"xWG4z96MUe",
	"9ZCLJPnZ9G",
	"xtrP9NjGYy",
	"T9MCkLW22n",
	"YoUB4RpNvE",
	"ipPRDRQ6fH",
	"mc3poBEmNp",
	"eGnG33mFaW",
}

func fetchGCG(gameID string) (string, error) {
	body := fmt.Sprintf(`{"gameId":"%s"}`, gameID)
	resp, err := http.Post(
		"https://woogles.io/api/game_service.GameMetadataService/GetGCG",
		"application/json",
		strings.NewReader(body),
	)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	var result struct {
		GCG string `json:"gcg"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return "", fmt.Errorf("unmarshal: %w (body: %s)", err, string(data[:min(200, len(data))]))
	}
	return result.GCG, nil
}

func bonusSummary(s *pb.PuzzleStats) string {
	parts := []string{}
	if s.GetTwsCovered() > 0 {
		parts = append(parts, fmt.Sprintf("TWS×%d", s.GetTwsCovered()))
	}
	if s.GetDwsCovered() > 0 {
		parts = append(parts, fmt.Sprintf("DWS×%d", s.GetDwsCovered()))
	}
	if s.GetTlsCovered() > 0 {
		parts = append(parts, fmt.Sprintf("TLS×%d", s.GetTlsCovered()))
	}
	if s.GetDlsCovered() > 0 {
		parts = append(parts, fmt.Sprintf("DLS×%d", s.GetDlsCovered()))
	}
	if len(parts) == 0 {
		return "none"
	}
	return strings.Join(parts, " ")
}

func tagNames(tags []pb.PuzzleTag) []string {
	out := make([]string, len(tags))
	for i, t := range tags {
		out[i] = t.String()
	}
	return out
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	zerolog.SetGlobalLevel(zerolog.Disabled)
	conf := config.DefaultConfig()

	req := &pb.PuzzleGenerationRequest{
		EquityMargin: 10,
		ScoreMargin:  10,
		Buckets: []*pb.PuzzleBucket{
			{Includes: []pb.PuzzleTag{}, Excludes: []pb.PuzzleTag{}},
		},
	}
	if err := puzzles.InitializePuzzleGenerationRequest(req); err != nil {
		fmt.Fprintln(os.Stderr, "init req:", err)
		os.Exit(1)
	}

	totalPuzzles := 0
	tagCounts := map[string]int{}

	for _, gid := range gameIDs {
		gcgStr, err := fetchGCG(gid)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [%s] fetch error: %v\n", gid, err)
			continue
		}

		// Write GCG to temp file (gcgio.ParseGCG takes a path)
		f, err := os.CreateTemp("", "woogles-*.gcg")
		if err != nil {
			fmt.Fprintln(os.Stderr, "tempfile:", err)
			continue
		}
		f.WriteString(gcgStr)
		f.Close()

		gh, err := gcgio.ParseGCG(conf, f.Name())
		os.Remove(f.Name())
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [%s] parse error: %v\n", gid, err)
			continue
		}

		// Determine lexicon from GCG history
		lexicon := gh.Lexicon
		if lexicon == "" {
			lexicon = "CSW24"
		}

		rules, err := game.NewBasicGameRules(conf, lexicon, board.CrosswordGameLayout, "english", game.CrossScoreAndSet, game.VarClassic)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [%s] rules error: %v\n", gid, err)
			continue
		}
		g, err := game.NewFromHistory(gh, rules, 0)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [%s] history error: %v\n", gid, err)
			continue
		}
		pzls, err := puzzles.CreatePuzzlesFromGame(conf, 999, g, req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [%s] puzzle error: %v\n", gid, err)
			continue
		}

		fmt.Printf("\n━━━ Game %s  (%d puzzles) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", gid, len(pzls))
		totalPuzzles += len(pzls)
		for _, pz := range pzls {
			s := pz.Stats
			ans := pz.Answer
			dir := "→"
			if ans.Direction == pb.GameEvent_VERTICAL {
				dir = "↓"
			}
			fmt.Printf("  turn%3d  %-18s %s%-3d  score=%3d  eq_adv=%+6.2f  score_adv=%+4d",
				pz.TurnNumber,
				ans.PlayedTiles,
				dir, ans.Column,
				s.GetScore(),
				s.GetEquityAdvantage(),
				s.GetScoreAdvantage(),
			)
			fmt.Printf("  words=%d  maxCross=%d", s.GetWordsFormed(), s.GetMaxCrossWordLength())
			if s.GetBonusSquaresCovered() > 0 {
				fmt.Printf("  bonus=%s", bonusSummary(s))
			}
			if s.GetLongestHookedWordLength() > 0 {
				fmt.Printf("  hook=%d", s.GetLongestHookedWordLength())
			}
			if s.GetLongestExtendedWordLength() > 0 {
				fmt.Printf("  ext=%d", s.GetLongestExtendedWordLength())
			}
			if s.GetMaxPlayedThroughTileScore() > 0 {
				fmt.Printf("  thruScore=%d", s.GetMaxPlayedThroughTileScore())
			}
			if s.GetMaxFreshTileFaceValueOnLetterBonusWithCrossword() > 0 {
				fmt.Printf("  DLS/TLS×cross=%d", s.GetMaxFreshTileFaceValueOnLetterBonusWithCrossword())
			}
			fmt.Printf("  rack_max=%d\n", s.GetMaxRackTileScore())
			fmt.Printf("           tags: %s\n", strings.Join(tagNames(pz.Tags), " "))
			for _, t := range pz.Tags {
				tagCounts[t.String()]++
			}
		}
	}

	fmt.Printf("\n━━━ SUMMARY: %d games, %d total puzzles ━━━━━━━━━━━━━━━━━━━━━━━━\n", len(gameIDs), totalPuzzles)
	fmt.Println("Tag distribution:")
	for tag, count := range tagCounts {
		fmt.Printf("  %-30s %d\n", tag, count)
	}
}
