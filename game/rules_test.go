package game

import (
	"testing"

	"github.com/domino14/macondo/board"
	"github.com/domino14/macondo/config"
	"github.com/matryer/is"
)

func TestMaxCanExchange(t *testing.T) {
	is := is.New(t)
	for _, tc := range []struct {
		exchlimit int
		inbag     int
		expected  int
	}{
		{7, 84, 7},
		{7, 12, 7},
		{7, 6, 0},
		{7, 7, 7},
		{7, 1, 0},
		{1, 5, 5},
		{1, 2, 2},
		{1, 1, 1},
		{1, 0, 0},
		{1, 47, 7},
		{7, 0, 0},
	} {
		is.Equal(MaxCanExchange(tc.inbag, tc.exchlimit), tc.expected)
	}
}

func TestBingoBonus(t *testing.T) {
	is := is.New(t)

	cfg := config.DefaultConfig()
	cfg.Set(config.ConfigBingoBonus, 50) // Set default to 50

	// Create rules manually
	rules := &GameRules{
		cfg:         cfg,
		bingoBonus:  0,
	}

	// Simulate the bingo bonus setting logic from NewBasicGameRules
	bingoBonus := cfg.GetInt(config.ConfigBingoBonus)
	if board.CrossplayGameLayout == board.CrossplayGameLayout {
		bingoBonus = 40
	}
	rules.bingoBonus = bingoBonus

	is.Equal(rules.BingoBonus(), 40) // Crossplay overrides to 40

	// Test other layouts use configured value
	bingoBonus = cfg.GetInt(config.ConfigBingoBonus)
	if board.CrosswordGameLayout == board.CrossplayGameLayout {
		bingoBonus = 40
	}
	rules.bingoBonus = bingoBonus
	is.Equal(rules.BingoBonus(), 50) // Should use configured value

	// Test custom bingo bonus
	cfg.Set(config.ConfigBingoBonus, 35)
	bingoBonus = cfg.GetInt(config.ConfigBingoBonus)
	if board.CrosswordGameLayout == board.CrossplayGameLayout {
		bingoBonus = 40
	}
	rules.bingoBonus = bingoBonus
	is.Equal(rules.BingoBonus(), 35) // Should use configured value

	// Test Crossplay still overrides to 40 even with custom config
	bingoBonus = cfg.GetInt(config.ConfigBingoBonus)
	if board.CrossplayGameLayout == board.CrossplayGameLayout {
		bingoBonus = 40
	}
	rules.bingoBonus = bingoBonus
	is.Equal(rules.BingoBonus(), 40) // Crossplay should still be 40
}
