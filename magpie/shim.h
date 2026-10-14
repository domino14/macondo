// The flat C API Macondo calls MAGPIE through. Every function exists twice,
// mgs15_* (BOARD_DIM 15) and mgs21_* (BOARD_DIM 21), compiled from shim_impl.h
// by shim_b15.c and shim_b21.c; the Go side picks a family per board size.
//
// Nothing here calls back into Go, and every buffer is caller-owned, so a call
// is one cgo crossing with no allocation on the hot path.
#ifndef MACONDO_MAGPIE_SHIM_H
#define MACONDO_MAGPIE_SHIM_H

#include <stdint.h>

// The largest board either family handles; mg_move.tiles is sized for it.
#define MG_MAX_DIM 21

// MAGPIE's game_event_t values for the moves the shim records.
#define MG_MOVE_PLACEMENT 1
#define MG_MOVE_PASS 3
#define MG_MOVE_EXCHANGE 5

// MAGPIE's move_record_t.
#define MG_RECORD_ALL 0
#define MG_RECORD_BEST 1
#define MG_RECORD_WITHIN_X 2

// MAGPIE's move_sort_t.
#define MG_SORT_EQUITY 0
#define MG_SORT_SCORE 1

#define MG_BACKUP_OFF 0
#define MG_BACKUP_STACK 1
#define MG_BACKUP_SINGLE 2

#define MG_VARIANT_CLASSIC 1
#define MG_VARIANT_WORDSMOG 2

// A move, independent of BOARD_DIM. Scores and equities are MAGPIE Equity
// (millipoints). tiles uses Macondo's and MAGPIE's shared encoding: 0 is a
// played-through square, letters 1..n, a designated blank has bit 0x80. For an
// exchange, tiles holds the exchanged tiles (0 = blank).
typedef struct mg_move {
  int32_t score;
  int32_t equity;
  uint8_t type;
  uint8_t row;
  uint8_t col;
  uint8_t vertical;
  uint8_t tiles_played;
  uint8_t tiles_length;
  uint8_t tiles[MG_MAX_DIM];
} mg_move;

typedef struct mg_data mg_data;
typedef struct mg_pos mg_pos;

#define MG_DECLARE_API(P)                                                      \
  /* Lexicon data shared read-only by every position made from it. kwg and  \
   * klv are the contents of a .kwg (or WordSmog .kad) and a .klv2 file.      \
   * ld_csv is a MAGPIE-format distribution (letter,lower,count,score,vowel   \
   * per line) and layout a MAGPIE layout file's contents. wmp_path may be    \
   * NULL; build_wmp then builds the word map in memory (classic only).       \
   * rit_path, a rack info table (needs a word map), may be NULL; it is       \
   * memory-mapped. On failure returns NULL with a message in err. */         \
  mg_data *P##data_create(const uint8_t *kwg, int64_t kwg_len,                 \
                          const uint8_t *klv, int64_t klv_len,                 \
                          const char *wmp_path, int build_wmp,                 \
                          const char *rit_path, const char *ld_csv,            \
                          const char *layout, int variant, int bingo_bonus,    \
                          char *err, int err_len);                             \
  /* Builds the data's rack info table and writes it to path. Returns 0, or   \
   * -1 with a message in err. */                                             \
  int P##data_write_rit(const mg_data *d, const char *path, int threads,       \
                        char *err, int err_len);                               \
  void P##data_destroy(mg_data *d);                                            \
  int P##data_has_wmp(const mg_data *d);                                       \
  /* Leave value of the multiset, in millipoints (0 with no leaves). */      \
  int32_t P##data_leave_value(const mg_data *d, const uint8_t *tiles, int n);  \
                                                                               \
  /* A position: board, cross-sets, racks and bag. Not thread-safe; one per   \
   * worker. move_capacity bounds MG_RECORD_ALL. */                           \
  mg_pos *P##pos_create(mg_data *d, int move_capacity, uint64_t seed);         \
  void P##pos_destroy(mg_pos *p);                                              \
  void P##pos_copy(mg_pos *dst, const mg_pos *src);                            \
  /* Sets the whole position and recomputes cross-sets and anchors.           \
   * squares is dim*dim row-major (0 = empty). Returns 0, or -1 if a tile is  \
   * not available in the distribution (message in err). */                   \
  int P##pos_set(mg_pos *p, const uint8_t *squares, const uint8_t *rack0,      \
                 int n0, const uint8_t *rack1, int n1, const uint8_t *bag,     \
                 int nbag, int on_turn, int32_t score0, int32_t score1,        \
                 int scoreless_turns, char *err, int err_len);                 \
  /* Replaces both racks, the bag and the player on turn. */                 \
  void P##pos_set_racks_and_bag(mg_pos *p, const uint8_t *rack0, int n0,       \
                                const uint8_t *rack1, int n1,                  \
                                const uint8_t *bag, int nbag, int on_turn);    \
  void P##pos_set_on_turn(mg_pos *p, int on_turn);                             \
  /* Switches the data (leaves, word map) the position generates with. The    \
   * new data must have the same word graph, distribution and layout, and     \
   * the old data must outlive the position. */                               \
  void P##pos_set_data(mg_pos *p, mg_data *d);                                 \
  /* What play backs up for unplay: MG_BACKUP_OFF, MG_BACKUP_STACK (every    \
   * play, any depth) or MG_BACKUP_SINGLE (only the last play). Backups       \
   * already held are kept. */                                                \
  void P##pos_set_backup_mode(mg_pos *p, int mode);                            \
  int P##pos_backup_depth(const mg_pos *p);                                    \
                                                                               \
  /* Generates moves for the player on turn into out (at most out_cap, which  \
   * is also how many MG_RECORD_ALL keeps: the best out_cap). Returns the     \
   * number written. margin is the equity window for MG_RECORD_WITHIN_X, in   \
   * millipoints. */                                                          \
  int P##gen(mg_pos *p, int record, int sort, int32_t margin, mg_move *out,    \
             int out_cap);                                                     \
                                                                               \
  /* Plays m for the player on turn. The tiles that player draws are given    \
   * in drawn (the caller owns the bag's randomness); they are taken out of   \
   * the bag. An exchange's tiles go back into the bag. Backs up first when   \
   * backups are on. Returns 0, or -1 if a drawn tile isn't in the bag. */    \
  int P##play(mg_pos *p, const mg_move *m, const uint8_t *drawn, int ndrawn);  \
  /* Undoes the last backed-up play. Returns 0, or -1 with no backup. */      \
  int P##unplay(mg_pos *p);                                                    \
  /* Undoes every backed-up play, back to the oldest backup, and anything     \
   * played since without one. Returns 0, or -1 with no backup. */            \
  int P##reset_to_first(mg_pos *p);                                            \
                                                                               \
  /* Copies out the board: letters (dim*dim), and per direction (0 = across   \
   * plays, 1 = down plays) cross-sets and cross-scores (2*dim*dim each,      \
   * direction-major), and anchors (2*dim*dim). Any pointer may be NULL. */   \
  void P##pos_read_board(const mg_pos *p, uint8_t *letters,                    \
                         uint64_t *cross_sets, int32_t *cross_scores,          \
                         uint8_t *anchors);                                    \
  void P##pos_read_racks(const mg_pos *p, uint8_t *counts0, uint8_t *counts1,  \
                         int *bag_count);

MG_DECLARE_API(mgs15_)
MG_DECLARE_API(mgs21_)

#endif
