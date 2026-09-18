#!/usr/bin/env bash
#
# Run the inference-vs-no-inference autoplay experiment.
#
# Updates a macondo checkout to the experiment branch, builds it, and starts the
# run detached so it survives losing the terminal. Everything it needs beyond the
# repo is checked for up front and reported by name, because the run takes days
# and a missing file should not surface an hour in.
#
#   ./run-inference-experiment.sh                 # run it
#   ./run-inference-experiment.sh --dry-run       # set up and stop before launching
#   ./run-inference-experiment.sh --pairs 3       # a short smoke test first
#   MACONDO_DIR=/scratch/macondo ./run-inference-experiment.sh
#
set -euo pipefail

BRANCH="${BRANCH:-claude/game-pair-autoplay-seed-wy2n1b}"
MACONDO_DIR="${MACONDO_DIR:-$HOME/macondo}"
CONFIG="${CONFIG:-infer-v-sim-5000.json}"
THREADS="${THREADS:-}"          # default: every logical CPU
PAIRS=""                        # override numGames, for a smoke test
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)  DRY_RUN=1; shift ;;
    --pairs)    PAIRS="$2"; shift 2 ;;
    --threads)  THREADS="$2"; shift 2 ;;
    --dir)      MACONDO_DIR="$2"; shift 2 ;;
    --branch)   BRANCH="$2"; shift 2 ;;
    -h|--help)  sed -n '2,14p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)          echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\n== %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
say "Checking prerequisites"
for tool in git go make; do
  command -v "$tool" >/dev/null || die "$tool is not installed or not on PATH"
done
echo "  git  $(git --version | awk '{print $3}')"
echo "  go   $(go version | awk '{print $3}')"

[ -d "$MACONDO_DIR" ] || die "no macondo checkout at $MACONDO_DIR
  Pass one with --dir /path/to/macondo, or clone it:
      git clone git@github.com:domino14/macondo.git $MACONDO_DIR"
cd "$MACONDO_DIR"
git rev-parse --git-dir >/dev/null 2>&1 || die "$MACONDO_DIR is not a git repository"
git remote get-url origin >/dev/null 2>&1 || die "$MACONDO_DIR has no 'origin' remote to fetch from.
  Add one:  git -C $MACONDO_DIR remote add origin git@github.com:domino14/macondo.git"

# Refuse to touch a dirty tree rather than checking out over someone's work.
if ! git diff --quiet || ! git diff --cached --quiet; then
  git status --short --untracked-files=no
  die "the working tree at $MACONDO_DIR has uncommitted changes (shown above).
  Commit or stash them first; this script will not check out over them."
fi

# ------------------------------------------------------------------- update
say "Updating to $BRANCH"
git fetch origin --prune
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
else
  git checkout -b "$BRANCH" --track "origin/$BRANCH"
fi
echo "  now at $(git rev-parse --short HEAD)  $(git log -1 --format=%s)"

# -------------------------------------------------------------------- build
say "Building"
make macondo_shell
[ -x ./bin/shell ] || die "build finished but ./bin/shell is missing"

# ----------------------------------------------------------------- the data
# data/lexica is gitignored, so the lexicon never arrives with a checkout. The
# run needs both the word graph and the leave values for its lexicon.
say "Checking lexicon data"
LEX="$(sed -n 's/.*"lexicon"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$CONFIG" 2>/dev/null || true)"
LEX="${LEX:-NWL23}"
missing=""
for f in "data/lexica/gaddag/$LEX.kwg" "data/lexica/gaddag/$LEX.klv2"; do
  [ -f "$f" ] || missing="$missing\n    $MACONDO_DIR/$f"
done
[ -z "$missing" ] || die "the $LEX lexicon files are missing:$(printf "$missing")
  data/lexica is gitignored, so these do not come with the repository.
  Copy them from a working macondo install, or pick a lexicon you do have
  by editing \"lexicon\" in $CONFIG. Present here:
$(ls data/lexica/gaddag/*.kwg 2>/dev/null | sed 's/^/    /' || echo '    (none)')"
echo "  $LEX: word graph and leaves both present"

# --------------------------------------------------------------- the config
# Untracked in the repo, so write it if this is a fresh checkout. An existing
# file is left alone, so local edits survive re-running this script.
if [ ! -f "$CONFIG" ]; then
  say "Writing $CONFIG (was not present)"
  cat > "$CONFIG" <<'JSON'
{
  "description": "Inference (tau 0.05) vs no inference, 5-ply sims both sides, 5000 game pairs",
  "experimentId": "infer-v-sim-5000",
  "lexicon": "NWL23",
  "letterDistribution": "English",
  "numGames": 5000,
  "threads": 128,
  "outputDir": "./experiments",
  "block": true,
  "gamePairs": true,
  "seed": 20260911,

  "player1": {
    "botCode": "SIMMING_INFER_BOT_NO_EG",
    "minSimPlies": 5,
    "simThreads": 1,
    "inferenceTau": 0.05,
    "inferenceBudget": 200
  },
  "player2": {
    "botCode": "SIMMING_BOT_NO_EG",
    "minSimPlies": 5,
    "simThreads": 1
  }
}
JSON
else
  say "Using the $CONFIG already here"
fi

# ------------------------------------------------------------------ sizing
# Each game is single-threaded on purpose -- that is what makes a paired run
# reproducible -- so the parallelism comes from running many games at once. One
# game per logical CPU measured fastest.
[ -n "$THREADS" ] || THREADS="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
NUMGAMES="$(sed -n 's/.*"numGames"[[:space:]]*:[[:space:]]*\([0-9]*\).*/\1/p' "$CONFIG")"
[ -n "$PAIRS" ] && NUMGAMES="$PAIRS"
EXPID="$(sed -n 's/.*"experimentId"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$CONFIG")"
[ -n "$PAIRS" ] && EXPID="${EXPID}-smoke"

# Measured on a 16-thread desktop: 11.4 pairs/hour at 5 plies with a 200-leaf
# inference budget. Scales roughly with core count.
EST_HOURS="$(awk -v p="$NUMGAMES" -v t="$THREADS" 'BEGIN{printf "%.1f", p/(11.4*t/16)}')"

say "Plan"
cat <<EOF
  branch        $BRANCH @ $(git rev-parse --short HEAD)
  config        $CONFIG
  pairs         $NUMGAMES   (= $((NUMGAMES * 2)) games; numGames counts PAIRS)
  threads       $THREADS concurrent games, 1 sim thread + 1 inference thread each
  lexicon       $LEX
  experiment    $EXPID
  estimate      ~$EST_HOURS hours
  output        $MACONDO_DIR/experiments/games-$EXPID.txt
                $MACONDO_DIR/experiments/$EXPID.txt
EOF

if [ "$DRY_RUN" = "1" ]; then
  say "--dry-run: everything is ready, not launching"
  exit 0
fi

# --------------------------------------------------------------------- run
# nohup, because this runs for days and a dropped ssh session should not take it
# down. Overrides go on the command line so the config file stays as written.
LOG="$MACONDO_DIR/$EXPID.log"
say "Starting (log: $LOG)"
nohup sh -c "printf 'autoplay $CONFIG -threads $THREADS -numgames $NUMGAMES -experimentid $EXPID\nexit\n' | ./bin/shell" \
  > "$LOG" 2>&1 &
PID=$!
sleep 5
kill -0 "$PID" 2>/dev/null || die "it exited immediately -- see $LOG"

cat <<EOF

  Running as pid $PID.

  Progress:   wc -l $MACONDO_DIR/experiments/games-$EXPID.txt   # 2 lines per pair
  Watch:      tail -f $LOG
  Stop:       kill $PID

  Results (partial results are valid at any point):
    ./bin/shell
    autoanalyze experiments/games-$EXPID.txt              # who won, paired
    autoanalyze experiments/games-$EXPID.txt -divergence  # where the bots differed
    autoanalyze experiments/$EXPID.txt -inference         # how good the inference was
EOF
