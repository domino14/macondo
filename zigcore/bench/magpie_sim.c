// Times MAGPIE's sim on a file of CGPs, one per line (Macondo's form; the
// opponent's rack and anything from " lex " on are dropped): the 15 best
// plays by static equity, round-robin, two plies, `per_play` rollouts a
// play, on `threads` threads, by wall clock around MAGPIE's own commands
// (MAGPIE's printed sim time leaves part of the work out). Built and run
// by magpie_sim.sh.
//
// Usage: magpie_sim <positions> <data path> <true|false (rack info table)>
//        <threads> <rollouts per play>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "impl/config.h"
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
  if (argc < 6) return 2;
  const int threads = atoi(argv[4]);
  const int per_play = atoi(argv[5]);
  ErrorStack *es = error_stack_create();
  ConfigArgs args = {.data_paths = argv[2], .settings_filename = NULL, .use_wmp = true};
  Config *config = config_create(&args, es);
  if (!error_stack_is_empty(es)) { error_stack_print_and_reset(es); return 1; }
  char set[512];
  snprintf(set, sizeof set,
           "set -lex NWL23 -ld english -wmp true -rit %s -wit false -pat none -threads %d "
           "-numplays 15 -plies 2 -sr rr -threshold none -iterations %d -minplayiterations 1 "
           "-seed 1 -pfrequency 0 -savesettings false -printboards false",
           argv[3], threads, 15 * per_play);
  run(config, set);
  FILE *f = fopen(argv[1], "r");
  char line[4096];
  static char cmds[256][4200];
  int n = 0;
  while (fgets(line, sizeof line, f) && n < 256) {
    char *p = strstr(line, " lex ");
    if (p) *p = 0;
    snprintf(cmds[n++], sizeof cmds[0], "cgp %s", line);
  }
  // Loading and generating aside: the time of the sims alone.
  double gen_time = 0, total = 0;
  for (int i = 0; i < n; i++) {
    run(config, cmds[i]);
    double t = now();
    run(config, "gen");
    gen_time += now() - t;
    t = now();
    run(config, "sim");
    total += now() - t;
  }
  (void)gen_time;
  printf("MAGPIE (rack info table %s, %d threads): %d positions, %.2f us/rollout by wall clock\n",
         argv[3], threads, n, total * 1e6 / ((double)n * 15 * per_play));
  return 0;
}
