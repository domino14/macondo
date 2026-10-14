// The shim's implementation, compiled once per board size: shim_b15.c and
// shim_b21.c define BOARD_DIM and MGS(name) (the mgs15_/mgs21_ prefix) and
// include this. Keep it free of anything that isn't parameterized by those two.
#include "shim.h"

#include "def/game_defs.h"
#include "def/game_history_defs.h"
#include "def/move_defs.h"
#include "def/players_data_defs.h"
#include "ent/bag.h"
#include "ent/board.h"
#include "ent/board_layout.h"
#include "ent/equity.h"
#include "ent/game.h"
#include "ent/klv.h"
#include "ent/kwg.h"
#include "ent/letter_distribution.h"
#include "ent/move.h"
#include "ent/player.h"
#include "ent/players_data.h"
#include "ent/rack.h"
#include "ent/rack_info_table.h"
#include "ent/wmp.h"
#include "impl/gameplay.h"
#include "impl/move_gen.h"
#include "impl/rack_info_table_maker.h"
#include "impl/wmp_maker.h"
#include "util/io_util.h"
#include "util/string_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if BOARD_DIM > MG_MAX_DIM
#error "BOARD_DIM exceeds MG_MAX_DIM"
#endif

struct mg_data {
  PlayersData *players_data;
  LetterDistribution *ld;
  BoardLayout *layout;
  int variant;
  int bingo_bonus;
};

// What a play changes, saved for unplay. MAGPIE's own game backups are
// capped at MAX_SEARCH_DEPTH; Macondo's stack has no fixed depth.
typedef struct mgs_backup {
  Board *board;
  Bag *bag;
  Rack racks[2];
  Equity scores[2];
  int on_turn;
  int scoreless_turns;
  game_end_reason_t end_reason;
} mgs_backup;

struct mg_pos {
  mg_data *data;
  Game *game;
  MoveList *move_list;
  int backup_mode;
  int backup_depth;
  int backup_cap;
  mgs_backup *backups;
};

static void mgs_set_err(char *err, int err_len, const char *msg) {
  if (err && err_len > 0) {
    snprintf(err, (size_t)err_len, "%s", msg ? msg : "unknown error");
  }
}

// Moves the error stack's message into err and resets it. Returns true if
// there was one.
static bool mgs_take_error(ErrorStack *es, char *err, int err_len) {
  if (error_stack_is_empty(es)) {
    return false;
  }
  char *msg = error_stack_get_string_and_reset(es);
  mgs_set_err(err, err_len, msg);
  free(msg);
  return true;
}

// Reads n little-endian uint32 KWG nodes from buf.
static void mgs_read_nodes(KWG *kwg, const uint8_t *buf, size_t n) {
  kwg_allocate_nodes(kwg, n);
  memcpy(kwg->nodes, buf, n * sizeof(uint32_t));
  for (size_t i = 0; i < n; i++) {
    kwg->nodes[i] = le32toh(kwg->nodes[i]);
  }
}

static KWG *mgs_kwg_from_bytes(const uint8_t *buf, int64_t len) {
  KWG *kwg = calloc_or_die(1, sizeof(KWG));
  kwg->name = string_duplicate("macondo");
  mgs_read_nodes(kwg, buf, (size_t)len / sizeof(uint32_t));
  return kwg;
}

// Parses a .klv2: a u32 node count, the KWG nodes, a u32 leave count and
// that many float32 leave values. As klv_load, but from memory.
static KLV *mgs_klv_from_bytes(const uint8_t *buf, int64_t len, char *err,
                               int err_len) {
  uint32_t kwg_size, number_of_leaves;
  if (len < 4) {
    goto truncated;
  }
  memcpy(&kwg_size, buf, 4);
  kwg_size = le32toh(kwg_size);
  const int64_t leaves_at = 4 + (int64_t)kwg_size * 4;
  if (len < leaves_at + 4) {
    goto truncated;
  }
  memcpy(&number_of_leaves, buf + leaves_at, 4);
  number_of_leaves = le32toh(number_of_leaves);
  if (len < leaves_at + 4 + (int64_t)number_of_leaves * 4) {
    goto truncated;
  }
  KLV *klv = calloc_or_die(1, sizeof(KLV));
  klv->name = string_duplicate("macondo");
  klv->kwg = kwg_create_empty();
  mgs_read_nodes(klv->kwg, buf + 4, kwg_size);
  klv->number_of_leaves = number_of_leaves;
  klv->leave_values = malloc_or_die(number_of_leaves * sizeof(Equity));
  const uint8_t *leaves = buf + leaves_at + 4;
  for (uint32_t i = 0; i < number_of_leaves; i++) {
    uint32_t bits;
    memcpy(&bits, leaves + 4 * (size_t)i, 4);
    bits = le32toh(bits);
    float f;
    memcpy(&f, &bits, 4);
    klv->leave_values[i] = double_to_equity((double)f);
  }
  klv->word_counts = calloc_or_die(kwg_size, sizeof(uint32_t));
  klv_count_words(klv, kwg_size);
  return klv;
truncated:
  mgs_set_err(err, err_len, "leave file is truncated");
  return NULL;
}

mg_data *MGS(data_create)(const uint8_t *kwg_buf, int64_t kwg_len,
                          const uint8_t *klv_buf, int64_t klv_len,
                          const char *wmp_path, int build_wmp,
                          const char *rit_path, const char *ld_csv,
                          const char *layout, int variant, int bingo_bonus,
                          char *err, int err_len) {
  ErrorStack *es = error_stack_create();
  mg_data *d = calloc_or_die(1, sizeof(mg_data));
  d->variant = variant;
  d->bingo_bonus = bingo_bonus;

  StringSplitter *ld_lines = split_string_by_newline(ld_csv, true);
  d->ld = calloc_or_die(1, sizeof(LetterDistribution));
  ld_create_internal("macondo", ld_lines, d->ld, es);
  string_splitter_destroy(ld_lines);
  if (mgs_take_error(es, err, err_len)) {
    goto fail;
  }

  StringSplitter *layout_lines = split_string_by_newline(layout, true);
  d->layout = board_layout_create();
  board_layout_parse_split_file(d->layout, "macondo", layout_lines, es);
  string_splitter_destroy(layout_lines);
  if (mgs_take_error(es, err, err_len)) {
    goto fail;
  }

  d->players_data = players_data_create(wmp_path != NULL || build_wmp);

  KWG *kwg = mgs_kwg_from_bytes(kwg_buf, kwg_len);
  players_data_set_data(d->players_data, PLAYERS_DATA_TYPE_KWG, 0, kwg);
  players_data_set_data(d->players_data, PLAYERS_DATA_TYPE_KWG, 1, kwg);
  players_data_set_is_shared(d->players_data, PLAYERS_DATA_TYPE_KWG, true);

  KLV *klv = mgs_klv_from_bytes(klv_buf, klv_len, err, err_len);
  if (!klv) {
    goto fail;
  }
  players_data_set_data(d->players_data, PLAYERS_DATA_TYPE_KLV, 0, klv);
  players_data_set_data(d->players_data, PLAYERS_DATA_TYPE_KLV, 1, klv);
  players_data_set_is_shared(d->players_data, PLAYERS_DATA_TYPE_KLV, true);

  WMP *wmp = NULL;
  if (wmp_path) {
    wmp = calloc_or_die(1, sizeof(WMP));
    wmp_load_from_filename(wmp, wmp_path, wmp_path, es);
    if (mgs_take_error(es, err, err_len)) {
      wmp_destroy(wmp);
      goto fail;
    }
  } else if (build_wmp && variant == MG_VARIANT_CLASSIC) {
    wmp = make_wmp_from_kwg(kwg, d->ld, 0);
  }
  if (wmp) {
    players_data_set_data(d->players_data, PLAYERS_DATA_TYPE_WMP, 0, wmp);
    players_data_set_data(d->players_data, PLAYERS_DATA_TYPE_WMP, 1, wmp);
    players_data_set_is_shared(d->players_data, PLAYERS_DATA_TYPE_WMP, true);
  }

  if (rit_path) {
    if (!wmp) {
      mgs_set_err(err, err_len, "a rack info table needs a word map");
      goto fail;
    }
    RackInfoTable *rit = calloc_or_die(1, sizeof(RackInfoTable));
    rit->instance_id = rack_info_table_next_instance_id();
    rack_info_table_load_mmap(rit, rit_path, rit_path, es);
    if (mgs_take_error(es, err, err_len)) {
      rack_info_table_destroy(rit);
      goto fail;
    }
    players_data_set_data(d->players_data, PLAYERS_DATA_TYPE_RIT, 0, rit);
    players_data_set_data(d->players_data, PLAYERS_DATA_TYPE_RIT, 1, rit);
    players_data_set_is_shared(d->players_data, PLAYERS_DATA_TYPE_RIT, true);
  }

  error_stack_destroy(es);
  return d;

fail:
  error_stack_destroy(es);
  MGS(data_destroy)(d);
  return NULL;
}

void MGS(data_destroy)(mg_data *d) {
  if (!d) {
    return;
  }
  players_data_destroy(d->players_data);
  if (d->layout) {
    board_layout_destroy(d->layout);
  }
  if (d->ld) {
    free(d->ld->name);
    free(d->ld);
  }
  free(d);
}

int MGS(data_write_rit)(const mg_data *d, const char *path, int threads,
                        char *err, int err_len) {
  const KLV *klv = players_data_get_klv(d->players_data, 0);
  const WMP *wmp = players_data_get_wmp(d->players_data, 0);
  if (!klv || !wmp) {
    mgs_set_err(err, err_len, "a rack info table needs leaves and a word map");
    return -1;
  }
  // Coverage [1, RACK_SIZE], as MAGPIE's own convert command builds it.
  RackInfoTable *rit = make_rack_info_table(klv, wmp, d->ld, threads, 1);
  ErrorStack *es = error_stack_create();
  rack_info_table_write_to_file(rit, path, es);
  rack_info_table_destroy(rit);
  const bool failed = mgs_take_error(es, err, err_len);
  error_stack_destroy(es);
  return failed ? -1 : 0;
}

int MGS(data_has_wmp)(const mg_data *d) {
  return players_data_get_data(d->players_data, PLAYERS_DATA_TYPE_WMP, 0) !=
         NULL;
}

int32_t MGS(data_leave_value)(const mg_data *d, const uint8_t *tiles, int n) {
  const KLV *klv = players_data_get_klv(d->players_data, 0);
  if (!klv || n == 0) {
    return 0;
  }
  Rack leave;
  rack_set_dist_size_and_reset(&leave, ld_get_size(d->ld));
  for (int i = 0; i < n; i++) {
    rack_add_letter(&leave, tiles[i]);
  }
  return klv_get_leave_value(klv, &leave);
}

mg_pos *MGS(pos_create)(mg_data *d, int move_capacity, uint64_t seed) {
  mg_pos *p = calloc_or_die(1, sizeof(mg_pos));
  p->data = d;
  GameArgs args = {
      .players_data = d->players_data,
      .board_layout = d->layout,
      .ld = d->ld,
      .game_variant = (game_variant_t)d->variant,
      .bingo_bonus = d->bingo_bonus,
      .seed = seed,
  };
  p->game = game_create(&args);
  p->move_list = move_list_create(move_capacity < 1 ? 1 : move_capacity);
  return p;
}

static void mgs_free_backups(mg_pos *p) {
  for (int i = 0; i < p->backup_cap; i++) {
    board_destroy(p->backups[i].board);
    bag_destroy(p->backups[i].bag);
  }
  free(p->backups);
  p->backups = NULL;
  p->backup_cap = 0;
  p->backup_depth = 0;
}

void MGS(pos_destroy)(mg_pos *p) {
  if (!p) {
    return;
  }
  mgs_free_backups(p);
  move_list_destroy(p->move_list);
  game_destroy(p->game);
  free(p);
}

void MGS(pos_copy)(mg_pos *dst, const mg_pos *src) {
  game_copy(dst->game, src->game);
  dst->backup_depth = 0;
}

static void mgs_backup_state(mg_pos *p) {
  if (p->backup_mode == MG_BACKUP_OFF) {
    return;
  }
  if (p->backup_mode == MG_BACKUP_SINGLE) {
    p->backup_depth = 0;
  }
  if (p->backup_depth == p->backup_cap) {
    const int cap = p->backup_cap ? 2 * p->backup_cap : 8;
    p->backups = realloc_or_die(p->backups, cap * sizeof(mgs_backup));
    for (int i = p->backup_cap; i < cap; i++) {
      p->backups[i].board = board_duplicate(game_get_board(p->game));
      p->backups[i].bag = bag_create(p->data->ld, 0);
    }
    p->backup_cap = cap;
  }
  mgs_backup *b = &p->backups[p->backup_depth++];
  const Game *game = p->game;
  board_copy(b->board, game_get_board(game));
  bag_copy(b->bag, game_get_bag(game));
  for (int i = 0; i < 2; i++) {
    const Player *player = game_get_player(game, i);
    rack_copy(&b->racks[i], player_get_rack(player));
    b->scores[i] = player_get_score(player);
  }
  b->on_turn = game_get_player_on_turn_index(game);
  b->scoreless_turns = game_get_consecutive_scoreless_turns(game);
  b->end_reason = game_get_game_end_reason(game);
}

void MGS(pos_set_backup_mode)(mg_pos *p, int mode) {
  p->backup_mode = mode;
  p->backup_depth = 0;
}

int MGS(pos_backup_depth)(const mg_pos *p) { return p->backup_depth; }

static void mgs_fill_rack(Rack *rack, const uint8_t *tiles, int n) {
  rack_reset(rack);
  for (int i = 0; i < n; i++) {
    rack_add_letter(rack, tiles[i]);
  }
}

static void mgs_set_bag(Bag *bag, const uint8_t *tiles, int n) {
  bag_set_to_tiles(bag, tiles, n);
}

int MGS(pos_set)(mg_pos *p, const uint8_t *squares, const uint8_t *rack0,
                 int n0, const uint8_t *rack1, int n1, const uint8_t *bag,
                 int nbag, int on_turn, int32_t score0, int32_t score1,
                 int scoreless_turns, char *err, int err_len) {
  Game *game = p->game;
  game_reset(game);
  Board *board = game_get_board(game);
  for (int row = 0; row < BOARD_DIM; row++) {
    for (int col = 0; col < BOARD_DIM; col++) {
      const MachineLetter ml = squares[row * BOARD_DIM + col];
      if (ml == ALPHABET_EMPTY_SQUARE_MARKER) {
        continue;
      }
      const MachineLetter unblanked = get_unblanked_machine_letter(ml);
      if (unblanked >= ld_get_size(p->data->ld)) {
        mgs_set_err(err, err_len, "board tile outside the distribution");
        return -1;
      }
      board_set_letter(board, row, col, ml);
      board_increment_tiles_played(board, 1);
    }
  }
  mgs_fill_rack(player_get_rack(game_get_player(game, 0)), rack0, n0);
  mgs_fill_rack(player_get_rack(game_get_player(game, 1)), rack1, n1);
  mgs_set_bag(game_get_bag(game), bag, nbag);
  player_set_score(game_get_player(game, 0), score0);
  player_set_score(game_get_player(game, 1), score1);
  game_set_starting_player_index(game, on_turn);
  game_set_consecutive_scoreless_turns(game, scoreless_turns);
  game_gen_all_cross_sets(game);
  board_update_all_anchors(board);
  p->backup_depth = 0;
  return 0;
}

void MGS(pos_set_racks_and_bag)(mg_pos *p, const uint8_t *rack0, int n0,
                                const uint8_t *rack1, int n1,
                                const uint8_t *bag, int nbag) {
  Game *game = p->game;
  mgs_fill_rack(player_get_rack(game_get_player(game, 0)), rack0, n0);
  mgs_fill_rack(player_get_rack(game_get_player(game, 1)), rack1, n1);
  mgs_set_bag(game_get_bag(game), bag, nbag);
}

void MGS(pos_set_on_turn)(mg_pos *p, int on_turn) {
  game_set_player_on_turn_index(p->game, on_turn);
}

static void mgs_move_out(const Move *m, mg_move *out) {
  out->score = move_get_score(m);
  out->equity = move_get_equity(m);
  out->type = (uint8_t)move_get_type(m);
  out->row = (uint8_t)move_get_row_start(m);
  out->col = (uint8_t)move_get_col_start(m);
  out->vertical = (uint8_t)(move_get_dir(m) == BOARD_VERTICAL_DIRECTION);
  out->tiles_played = (uint8_t)move_get_tiles_played(m);
  out->tiles_length = (uint8_t)move_get_tiles_length(m);
  const int n = out->tiles_length;
  for (int i = 0; i < n; i++) {
    out->tiles[i] = move_get_tile(m, i);
  }
}

static void mgs_move_in(const mg_move *in, Move *m) {
  move_set_type(m, (game_event_t)in->type);
  move_set_score(m, in->score);
  move_set_equity(m, in->equity);
  move_set_row_start(m, in->row);
  move_set_col_start(m, in->col);
  move_set_dir(m, in->vertical ? BOARD_VERTICAL_DIRECTION
                               : BOARD_HORIZONTAL_DIRECTION);
  move_set_tiles_played(m, in->tiles_played);
  move_set_tiles_length(m, in->tiles_length);
  for (int i = 0; i < in->tiles_length; i++) {
    move_set_tile(m, in->tiles[i], i);
  }
}

int MGS(gen)(mg_pos *p, int record, int sort, int32_t margin, mg_move *out,
             int out_cap) {
  MoveList *ml = p->move_list;
  const MoveGenArgs args = {
      .game = p->game,
      .move_list = ml,
      .move_record_type = (move_record_t)record,
      .move_sort_type = (move_sort_t)sort,
      .override_kwg = NULL,
      .eq_margin_movegen = margin,
      .target_equity = EQUITY_MAX_VALUE,
      .target_leave_size_for_exchange_cutoff = UNSET_LEAVE_SIZE,
  };
  generate_moves(&args);
  if (record != MG_RECORD_BEST) {
    // ALL and WITHIN_X leave a min-heap; hand moves out best-first.
    move_list_sort_moves(ml);
  }
  int n = move_list_get_count(ml);
  if (n > out_cap) {
    n = out_cap;
  }
  for (int i = 0; i < n; i++) {
    mgs_move_out(move_list_get_move(ml, i), &out[i]);
  }
  return n;
}

// Takes each drawn tile out of the bag and puts it on the rack.
static int mgs_draw(Game *game, int player_index, const uint8_t *drawn,
                    int ndrawn) {
  Bag *bag = game_get_bag(game);
  Rack *rack = player_get_rack(game_get_player(game, player_index));
  const int draw_index = game_get_player_draw_index(game, player_index);
  for (int i = 0; i < ndrawn; i++) {
    if (!bag_draw_letter(bag, drawn[i], draw_index)) {
      return -1;
    }
    rack_add_letter(rack, drawn[i]);
  }
  return 0;
}

int MGS(play)(mg_pos *p, const mg_move *m, const uint8_t *drawn, int ndrawn) {
  Game *game = p->game;
  const int player_index = game_get_player_on_turn_index(game);
  mgs_backup_state(p);
  if (m->type == MG_MOVE_EXCHANGE) {
    // MAGPIE's exchange draws from its own bag; Macondo picks the draw.
    Rack *rack = player_get_rack(game_get_player(game, player_index));
    for (int i = 0; i < m->tiles_played; i++) {
      rack_take_letter(rack, m->tiles[i]);
    }
    if (mgs_draw(game, player_index, drawn, ndrawn) != 0) {
      return -1;
    }
    Bag *bag = game_get_bag(game);
    const int draw_index = game_get_player_draw_index(game, player_index);
    for (int i = 0; i < m->tiles_played; i++) {
      bag_add_letter(bag, m->tiles[i], draw_index);
    }
    game_increment_consecutive_scoreless_turns(game);
    game_start_next_player_turn(game);
    return 0;
  }
  Move move;
  mgs_move_in(m, &move);
  // Places the tiles (taking them off the rack), updates cross-sets, scores
  // and passes the turn. MAGPIE's own backups are off; the shim's are above.
  play_move_without_drawing_tiles(&move, game);
  if (m->type == MG_MOVE_PLACEMENT) {
    return mgs_draw(game, player_index, drawn, ndrawn);
  }
  return 0;
}

int MGS(unplay)(mg_pos *p) {
  if (p->backup_depth == 0) {
    return -1;
  }
  const mgs_backup *b = &p->backups[--p->backup_depth];
  Game *game = p->game;
  board_copy(game_get_board(game), b->board);
  bag_copy(game_get_bag(game), b->bag);
  for (int i = 0; i < 2; i++) {
    Player *player = game_get_player(game, i);
    rack_copy(player_get_rack(player), &b->racks[i]);
    player_set_score(player, b->scores[i]);
  }
  game_set_player_on_turn_index(game, b->on_turn);
  game_set_consecutive_scoreless_turns(game, b->scoreless_turns);
  game_set_game_end_reason(game, b->end_reason);
  return 0;
}

void MGS(pos_read_board)(const mg_pos *p, uint8_t *letters,
                         uint64_t *cross_sets, int32_t *cross_scores,
                         uint8_t *anchors) {
  const Board *b = game_get_board(p->game);
  const int ci = board_get_cross_set_index(
      game_get_data_is_shared(p->game, PLAYERS_DATA_TYPE_KWG),
      game_get_player_on_turn_index(p->game));
  enum { N = BOARD_DIM * BOARD_DIM };
  for (int row = 0; row < BOARD_DIM; row++) {
    for (int col = 0; col < BOARD_DIM; col++) {
      const int i = row * BOARD_DIM + col;
      if (letters) {
        letters[i] = board_get_letter(b, row, col);
      }
      for (int dir = 0; dir < 2; dir++) {
        if (cross_sets) {
          cross_sets[dir * N + i] = board_get_cross_set(b, row, col, dir, ci);
        }
        if (cross_scores) {
          cross_scores[dir * N + i] = board_get_cross_score(b, row, col, dir, ci);
        }
        if (anchors) {
          anchors[dir * N + i] = board_get_anchor(b, row, col, dir);
        }
      }
    }
  }
}

void MGS(pos_read_racks)(const mg_pos *p, uint8_t *counts0, uint8_t *counts1,
                         int *bag_count) {
  const Rack *r0 = player_get_rack(game_get_player(p->game, 0));
  const Rack *r1 = player_get_rack(game_get_player(p->game, 1));
  const int size = ld_get_size(p->data->ld);
  for (int i = 0; i < size; i++) {
    if (counts0) {
      counts0[i] = (uint8_t)rack_get_letter(r0, i);
    }
    if (counts1) {
      counts1[i] = (uint8_t)rack_get_letter(r1, i);
    }
  }
  if (bag_count) {
    *bag_count = bag_get_letters(game_get_bag(p->game));
  }
}
