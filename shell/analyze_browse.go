package shell

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/domino14/macondo/gameanalysis"
)

// analyzeBrowse shows a table of stored analyses and optionally deletes one.
func (sc *ShellController) analyzeBrowse(cmd *shellcmd) (*Response, error) {
	store, err := sc.getAnalysisStore()
	if err != nil {
		return nil, fmt.Errorf("open analysis store: %w", err)
	}

	// Handle delete subcommand
	if deleteTarget := cmd.options.String("delete"); deleteTarget != "" {
		if !store.Exists(deleteTarget) {
			return msg(fmt.Sprintf("No analysis found with name '%s'.", deleteTarget)), nil
		}

		// Confirm unless -yes flag is set
		if !cmd.options.Bool("yes") {
			sc.showMessage(fmt.Sprintf("Delete analysis '%s'? [y/N]: ", deleteTarget))
			line, err := sc.l.Readline()
			if err != nil || strings.TrimSpace(strings.ToLower(line)) != "y" {
				return msg("Cancelled."), nil
			}
		}

		if err := store.Delete(deleteTarget); err != nil {
			return nil, fmt.Errorf("delete failed: %w", err)
		}
		return msg(fmt.Sprintf("Deleted analysis '%s'.", deleteTarget)), nil
	}

	// List analyses
	batchFilter := cmd.options.String("batch")
	limit, err := cmd.options.IntDefault("limit", 20)
	if err != nil {
		return nil, fmt.Errorf("invalid limit: %w", err)
	}

	var rows []storedRow
	if batchFilter != "" {
		stored, err := store.ListByBatch(batchFilter)
		if err != nil {
			return nil, fmt.Errorf("list failed: %w", err)
		}
		rows = toRows(stored)
	} else {
		stored, err := store.List(limit)
		if err != nil {
			return nil, fmt.Errorf("list failed: %w", err)
		}
		rows = toRows(stored)
	}

	if len(rows) == 0 {
		if batchFilter != "" {
			return msg(fmt.Sprintf("No analyses found for batch '%s'.", batchFilter)), nil
		}
		return msg("No analyses stored yet. Run 'analyze' after loading a game."), nil
	}

	return msg(formatBrowseTable(rows)), nil
}

type storedRow struct {
	Name            string
	PlayerInfo      string
	Lexicon         string
	MI0             string
	MI1             string
	BatchName       string
	AnalysisVersion int
	Date            time.Time
}

func toRows(stored []gameanalysis.StoredAnalysis) []storedRow {
	rows := make([]storedRow, 0, len(stored))
	for _, s := range stored {
		row := storedRow{
			Name:            s.Name,
			PlayerInfo:      s.PlayerInfo,
			Lexicon:         s.Lexicon,
			BatchName:       s.BatchName,
			AnalysisVersion: s.AnalysisVersion,
			Date:            s.CreatedAt,
			MI0:             "—",
			MI1:             "—",
		}

		// Extract mistake indices from stored JSON
		var partial struct {
			PlayerSummaries []struct {
				MistakeIndex float64 `json:"mistakeIndex"`
			} `json:"playerSummaries"`
		}
		if err := json.Unmarshal(s.ResultJSON, &partial); err == nil {
			if len(partial.PlayerSummaries) > 0 && partial.PlayerSummaries[0].MistakeIndex > 0 {
				row.MI0 = fmt.Sprintf("%.2f", partial.PlayerSummaries[0].MistakeIndex)
			}
			if len(partial.PlayerSummaries) > 1 && partial.PlayerSummaries[1].MistakeIndex > 0 {
				row.MI1 = fmt.Sprintf("%.2f", partial.PlayerSummaries[1].MistakeIndex)
			}
		}

		rows = append(rows, row)
	}
	return rows
}

func formatBrowseTable(rows []storedRow) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("%-40s  %-20s  %-8s  %-6s  %-6s  %-15s  %s\n",
		"Name", "Players", "Lexicon", "MI(0)", "MI(1)", "Batch", "Date"))
	sb.WriteString(strings.Repeat("-", 120))
	sb.WriteString("\n")

	for _, r := range rows {
		name := r.Name
		if len(name) > 40 {
			name = name[:37] + "..."
		}
		players := r.PlayerInfo
		if len(players) > 20 {
			players = players[:17] + "..."
		}
		batch := r.BatchName
		if batch == "" {
			batch = "—"
		}
		if len(batch) > 15 {
			batch = batch[:12] + "..."
		}

		vTag := ""
		if r.AnalysisVersion >= 2 {
			vTag = " ✓"
		}

		sb.WriteString(fmt.Sprintf("%-40s  %-20s  %-8s  %-6s  %-6s  %-15s  %s%s\n",
			name, players, r.Lexicon, r.MI0, r.MI1, batch,
			r.Date.Format("2006-01-02 15:04"), vTag))
	}

	sb.WriteString(fmt.Sprintf("\n%d analyses shown. ✓ = enriched (v2+)\n", len(rows)))
	sb.WriteString("Use 'analyze-browse -delete <name>' to delete an entry.\n")
	sb.WriteString("Use 'analyze-browse -batch <name>' to filter by batch.\n")

	return sb.String()
}

