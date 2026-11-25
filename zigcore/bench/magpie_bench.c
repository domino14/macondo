// Times MAGPIE's best static move (get_top_equity_move) on a file of CGPs,
// one per line (Macondo's form: only the first four fields are used), to
// compare with `zig build bench` on the same positions. Built and run by
// magpie_bench.sh.
//
// Usage: magpie_bench <positions> <MAGPIE data path> [true|false (word maps)]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "impl/config.h"
#include "impl/cgp.h"
#include "impl/gameplay.h"
#include "ent/game.h"
#include "ent/move.h"
#include "ent/thread_control.h"
#include "util/io_util.h"

static void run(Config *config, const char *cmd) {
  ErrorStack *es = error_stack_create();
  thread_control_set_status(config_get_thread_control(config), THREAD_CONTROL_STATUS_STARTED);
  config_load_command(config, cmd, es);
  config_execute_command(config, es);
  if (!error_stack_is_empty(es)) { error_stack_print_and_reset(es); exit(1); }
  error_stack_destroy(es);
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  const char *wmp = argc > 3 ? argv[3] : "true";
  ErrorStack *es = error_stack_create();
  ConfigArgs args = {.data_paths = argv[2], .settings_filename = NULL, .use_wmp = strcmp(wmp, "true") == 0};
  Config *config = config_create(&args, es);
  if (!error_stack_is_empty(es)) { error_stack_print_and_reset(es); return 1; }
  char set[256];
  snprintf(set, sizeof set, "set -lex NWL23 -ld english -wmp %s -threads 1 -s1 equity -s2 equity -savesettings false", wmp);
  run(config, set);

  FILE *f = fopen(argv[1], "r");
  char line[4096];
  Game *games[4096];
  int n = 0;
  while (fgets(line, sizeof line, f) && n < 4096) {
    // The board, racks, scores and zeros: the first four fields.
    char *p = line;
    for (int fields = 0; *p; p++) {
      if (*p == ' ' && ++fields == 4) { *p = 0; break; }
    }
    games[n] = config_game_create(config);
    game_load_cgp(games[n], line, es);
    if (!error_stack_is_empty(es)) { error_stack_print_and_reset(es); return 1; }
    n++;
  }
  MoveList *ml = move_list_create(1);
  double best = 1e30, sum = 0;
  for (int round = 0; round < 10; round++) {
    double t = now(), s = 0;
    for (int i = 0; i < n; i++) {
      const Move *m = get_top_equity_move(games[i], ml);
      s += equity_to_double(move_get_equity(m));
    }
    t = now() - t;
    if (round == 0) printf("round 1: %.2f us/position\n", t / n * 1e6);
    if (t < best) best = t;
    sum = s;
  }
  printf("MAGPIE wmp=%s: %d positions, best of 10: %.2f us/position, equity sum %.3f\n", wmp, n, best / n * 1e6, sum);
  return 0;
}
