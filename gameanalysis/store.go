package gameanalysis

import (
	"database/sql"
	"fmt"
	"path/filepath"
	"time"

	_ "modernc.org/sqlite"
)

const createTableSQL = `
CREATE TABLE IF NOT EXISTS analyses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  batch_name TEXT NOT NULL DEFAULT '',
  player_info TEXT NOT NULL DEFAULT '',
  lexicon TEXT NOT NULL DEFAULT '',
  analysis_version INTEGER NOT NULL DEFAULT 0,
  analyzer_version TEXT NOT NULL DEFAULT '',
  result_json TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_analyses_batch ON analyses(batch_name);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
`

// StoredAnalysis is a row from the analyses table.
type StoredAnalysis struct {
	ID              int
	Name            string
	BatchName       string
	PlayerInfo      string
	Lexicon         string
	AnalysisVersion int
	AnalyzerVersion string
	ResultJSON      []byte
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

// AnalysisStore is a thin wrapper around the local SQLite analysis database.
type AnalysisStore struct {
	db *sql.DB
}

// OpenStore opens (or creates) the analysis database at <dataPath>/analysis.db.
func OpenStore(dataPath string) (*AnalysisStore, error) {
	dbPath := filepath.Join(dataPath, "analysis.db")
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return nil, fmt.Errorf("open analysis db: %w", err)
	}
	db.SetMaxOpenConns(1) // SQLite is single-writer
	if _, err := db.Exec(createTableSQL); err != nil {
		db.Close()
		return nil, fmt.Errorf("create analyses table: %w", err)
	}
	return &AnalysisStore{db: db}, nil
}

// Close closes the database connection.
func (s *AnalysisStore) Close() error {
	return s.db.Close()
}

// Exists reports whether an analysis with the given name is stored.
func (s *AnalysisStore) Exists(name string) bool {
	var n int
	_ = s.db.QueryRow(`SELECT COUNT(*) FROM analyses WHERE name = ?`, name).Scan(&n)
	return n > 0
}

// Save inserts or replaces an analysis record.
func (s *AnalysisStore) Save(name, batchName, playerInfo, lexicon string, analysisVersion int, analyzerVersion string, resultJSON []byte) error {
	_, err := s.db.Exec(`
		INSERT INTO analyses (name, batch_name, player_info, lexicon, analysis_version, analyzer_version, result_json, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
		ON CONFLICT(name) DO UPDATE SET
			batch_name       = excluded.batch_name,
			player_info      = excluded.player_info,
			lexicon          = excluded.lexicon,
			analysis_version = excluded.analysis_version,
			analyzer_version = excluded.analyzer_version,
			result_json      = excluded.result_json,
			updated_at       = CURRENT_TIMESTAMP
	`, name, batchName, playerInfo, lexicon, analysisVersion, analyzerVersion, string(resultJSON))
	return err
}

// Get retrieves a single analysis by name. Returns sql.ErrNoRows if not found.
func (s *AnalysisStore) Get(name string) (*StoredAnalysis, error) {
	row := s.db.QueryRow(`
		SELECT id, name, batch_name, player_info, lexicon, analysis_version, analyzer_version, result_json, created_at, updated_at
		FROM analyses WHERE name = ?`, name)
	return scanAnalysis(row)
}

// List returns the N most recent analyses, or all if limit <= 0.
func (s *AnalysisStore) List(limit int) ([]StoredAnalysis, error) {
	q := `SELECT id, name, batch_name, player_info, lexicon, analysis_version, analyzer_version, result_json, created_at, updated_at
	      FROM analyses ORDER BY created_at DESC`
	if limit > 0 {
		q += fmt.Sprintf(" LIMIT %d", limit)
	}
	return s.queryAnalyses(q)
}

// ListByBatch returns all analyses belonging to a batch, ordered by created_at.
func (s *AnalysisStore) ListByBatch(batchName string) ([]StoredAnalysis, error) {
	return s.queryAnalyses(`
		SELECT id, name, batch_name, player_info, lexicon, analysis_version, analyzer_version, result_json, created_at, updated_at
		FROM analyses WHERE batch_name = ? ORDER BY created_at ASC`, batchName)
}

// Delete removes an analysis by name. Returns nil if it didn't exist.
func (s *AnalysisStore) Delete(name string) error {
	_, err := s.db.Exec(`DELETE FROM analyses WHERE name = ?`, name)
	return err
}

func (s *AnalysisStore) queryAnalyses(query string, args ...interface{}) ([]StoredAnalysis, error) {
	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []StoredAnalysis
	for rows.Next() {
		a, err := scanAnalysisRow(rows)
		if err != nil {
			return nil, err
		}
		results = append(results, *a)
	}
	return results, rows.Err()
}

func scanAnalysis(row *sql.Row) (*StoredAnalysis, error) {
	var a StoredAnalysis
	var resultJSON string
	if err := row.Scan(&a.ID, &a.Name, &a.BatchName, &a.PlayerInfo, &a.Lexicon,
		&a.AnalysisVersion, &a.AnalyzerVersion, &resultJSON, &a.CreatedAt, &a.UpdatedAt); err != nil {
		return nil, err
	}
	a.ResultJSON = []byte(resultJSON)
	return &a, nil
}

func scanAnalysisRow(rows *sql.Rows) (*StoredAnalysis, error) {
	var a StoredAnalysis
	var resultJSON string
	if err := rows.Scan(&a.ID, &a.Name, &a.BatchName, &a.PlayerInfo, &a.Lexicon,
		&a.AnalysisVersion, &a.AnalyzerVersion, &resultJSON, &a.CreatedAt, &a.UpdatedAt); err != nil {
		return nil, err
	}
	a.ResultJSON = []byte(resultJSON)
	return &a, nil
}
