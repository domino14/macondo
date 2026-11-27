# Macondo Zig core: plan

## Goal

A small, separate, fast engine core written in Zig that can play or analyze a
crossword game at high speed, and that speaks one line-based protocol (UCGI,
see `UCGI.md`) to any interface:

- the Macondo shell (stays in Go) and other Go tools,
- a new native or browser GUI,
- an in-browser analyzer for Woogles, built as multithreaded WASM.

UCGI is meant to become a shared standard, so that other engines (MAGPIE,
wolges) can speak it too and interfaces can swap engines.

The port is done module by module. Each module is checked against the Go
implementation before the next one starts, and the Go code stays in place
until its Zig replacement passes.

## Architecture

```
 Go shell / GUI / Woogles analyzer
          |  UCGI lines over stdio (native), postMessage (WASM),
          |  or a socket bridge (optional, later)
 +--------v-------------------------------------------------+
 | Adapters   stdio loop | WASM exports | socket bridge     |  thin; the only
 +----------------------------------------------------------+  layer that does I/O
 | Engine     UCGI parser, search control, stop/info,       |
 |            thread pool (native) / Web Worker pool (WASM) |
 +----------------------------------------------------------+
 | Compute    movegen, equity, sim, BAI, inference,         |  pure: no I/O,
 |            endgame, preendgame, autoplay, ML features    |  no std.Io
 +----------------------------------------------------------+
 | Data       KWG, KLV2, WMP, KAD, letter distributions,    |
 |            board layouts (loaded from bytes)             |
 +----------------------------------------------------------+
```

Rules for the layers:

- **Compute and data take no I/O.** Files are loaded by the adapters and handed
  in as byte slices. This keeps Zig's I/O-related breaking changes contained
  in the adapter layer, and it is what makes the WASM build possible.
- **Allocators are passed in**, never global.
- **Every search can be stopped** (an atomic flag checked in the hot loops)
  and reports progress through a callback the engine turns into `info` lines.
- **Determinism:** with one thread and a fixed seed, every search must give
  the same result as the Go code. This is what the cross-checks rely on.
  Matching Go bit for bit takes Go's own algorithms where they decide a
  result: its sort (pdqsort's order of equal elements), math/rand, the
  ChaCha8 stream of its frand, its amd64 `exp`/`log` and `Lgamma`. These
  live apart from the engine (see Source layout) and go once Go is no
  longer the reference (step 13).

### Source layout

`src/` is grouped by subsystem, as the Go code is by package:

| Folder | What |
|---|---|
| `src/` | `root.zig` (the module's exports), `ucgi.zig` (the engine) |
| `src/adapter/` | the native adapter: `main.zig`, `files.zig`, `bench.zig` |
| `src/data/` | `tiles.zig`, `kwg.zig`, `data.zig` (loading), `wmp.zig`, `wmp_maker.zig`, `rit.zig` |
| `src/board/` | `board.zig`, `cross_set.zig`, `move.zig`, `cgp.zig`, `position.zig` |
| `src/movegen/` | `movegen.zig`, `shadow.zig`, `wmp_movegen.zig`, `wordmap_gen.zig`, `leavemap.zig`, `carry.zig` |
| `src/equity/` | `equity.zig` |
| `src/game/` | `game.zig` (bag, turns, autoplay) |
| `src/montecarlo/` | `sim.zig`, `bai.zig`, `classic.zig` |
| `src/rangefinder/` | `infer.zig` |
| `src/endgame/` | `endgame.zig`, `ttable.zig`, `zobrist.zig`, `pathlists.zig`, `wordprune.zig` |
| `src/gocompat/` | Go-exact algorithms and their golden files: `gosort.zig`, `gorand.zig`, `gomath.zig`, `chacha8.zig` (frand's generator, which the bag draws with) |
| `src/testing/` | `testdata.zig` |

A folder is only grouping: Zig's privacy is per file (a declaration
without `pub` is private to its file) and its package boundary is the
module `build.zig` declares, so there is no folder-level `internal` as in
Go. New subsystems (pre-endgame, WordSmog, ML) get a folder of their own.

## What moves to Zig and what stays in Go

### Moves to Zig

| Area | Go packages today | Notes |
|---|---|---|
| Tiles and lexica | `word-golib` (`tilemapping`, `kwg`, `cache`, letter distributions) | Port the parts the engine uses: machine letters, racks, distributions, KWG reading. Go keeps `word-golib` for the shell. |
| Word maps | `wmp` (reader, lookup, maker, `movegen.go`, `bitrack.go`, `policy.go`) | The maker too, with Go's output byte for byte: word maps are tens of times the size of their KWGs and aren't published, so an engine with no Go beside it (the WASM build above all) makes its own. The engine makes one in memory when the file is missing; writing it to disk is the adapter's choice. |
| Rack info tables | none in Go: MAGPIE's `rack_info_table` and its maker | Ported from MAGPIE with exact leave values (MAGPIE rounds to thousandths), so results stay Go's: they only make the best-play search faster. Optional (`rack-tables`), made and saved on request; checked against MAGPIE's tables made from the same files. |
| WordSmog | `alphadawg`, `cross_set/alpha_cross_set.go`, `movegen/alpha_gen.go`, `ai/turnplayer/wordsmog.go` | `.kad` files are plain DAWG-only KWGs. |
| Board and rules | `board` (incl. layouts, cross-sets), `cross_set`, `zobrist`, the rules part of `game` (`rules.go`, `turn.go`, `pairedbag.go`, bag and racks, challenge rules) | Game history, GCG, display and backup stay in Go. |
| Moves | `move`, `tinymove`, `leavemap` | |
| Move generation | `movegen` (incl. shadow, row cache, wordmap gen, `wordprune`) | First big milestone. |
| Equity | `equity` (KLV2 reader, exhaustive leave, opening, endgame and preendgame adjustments, combined) | |
| Static players | `ai/turnplayer`, `ai/simplesimmer` | |
| Simulation | `montecarlo` (incl. `bai`, stopping conditions, paired stops, candidates, endgame leaves) | `montecarlo/stats` presentation (heatmaps, lanes) stays in Go; the core reports raw stats. |
| Inference | `rangefinder` | |
| Endgame | `endgame/negamax` (TT, ABDADA, PV split, heuristics, prefetch) | Assembly prefetch becomes `@prefetch`. Huge pages become an adapter option. |
| Preendgame | `preendgame` | |
| Best-move policy | the algorithm choice in `ai/bot/elite.go` (static / sim / preendgame / endgame by bag size) | Exposed as `go auto`, so the bot, the shell and the browser all play the same way. |
| Autoplay | the game loop of `automatic` (`game_runner.go`) | For fast self-play and data generation. Log analysis, corpora and divergence tools stay in Go. |
| ML features | `game/mlbytes.go`, `game/mlhelper.go` (vector building), `game/mllayout.go` | Feature extraction runs per move inside searches, so it belongs in the core. |
| Local ML inference | `game/mlort.go` (ONNX Runtime through cgo) | Later phase. Zig calls the ONNX Runtime C API directly (declare the needed functions by hand or use the `translate-c` package; `@cImport` was removed in Zig 0.17). Not available in the WASM build at first. |

### Stays in Go

- `shell`, `cmd/*` (the many lab and tooling commands), `config` (viper)
- `gcgio`, game history and display, `cgp` parsing for the shell side
  (the core gets its own CGP parser)
- `bot` (NATS, Woogles), `worker`, `cmd/lambda`, `analyzer`
- `gameanalysis`, `puzzles`: orchestration in Go, the heavy calls go to the core
- `explainer` (LLM)
- `triton` client, `pytorch`, `notebooks`, model downloading (`game/mlfetch.go`,
  `dataloaders`)
- `ai/bot/filters.go` (weaker bot levels: common words, findability). Revisit
  if bots need to run in the browser.
- `ai/externalengine`: becomes the UCGI client (it already runs an external
  engine binary per turn; generalize it to a persistent UCGI session).
- `playability`, `stats`: research tools; can call the core later.
- `wasm/` (Go WASM build): replaced by the Zig WASM build once it covers the
  analyzer.

### Neural network evaluation

The Go code builds feature vectors for each move and evaluates them with ONNX
Runtime (cgo) or Triton (gRPC). Plan:

1. **Until the Zig ML phase**, the core has no NN evaluation. The NN bot path
   keeps using the Go engine.
2. **Feature extraction moves to Zig** with byte-for-byte tests against the
   Go vectors (`game/mlalphabet_test.go`, `mllayout_test.go` give fixtures).
3. **The core gets an `Evaluator` interface** with these backends:
   - `ort`: ONNX Runtime C API linked from Zig (replaces cgo in Go for the
     core's own use).
   - `remote`: the core sends a batch of feature vectors to the interface
     over UCGI (`evalreq` / `evalresp`), and the Go side answers with Triton.
     Fine for one batched call per turn (ranking the top plays); too slow
     inside sims.
   - Later, optionally: Triton's HTTP (KServe v2) protocol straight from Zig,
     and `onnxruntime-web` through JS imports for WASM.

## Port order

Each step is done when its cross-check passes (see Testing). Check items off
as they land.

- [x] **0. Scaffolding.** `zigcore/` with `build.zig`, pinned Zig version, CI
      job, a `macondo-zig` binary that answers `ucgi` / `isready` / `quit`, and the
      Go cross-check harness skeleton (`zigcore/crosscheck`, a Go test
      package that starts the binary).
- [x] **1. Data and board.** Machine letters, letter distributions, racks,
      KWG reader, KLV2 reader, board layouts, board state, cross-sets and
      cross-scores (with anchors and extension sets), CGP parsing;
      `position cgp`, `load`, and the debug commands `crosssets` and
      `leave`. Check: every square identical to Go on positions from random
      games and on random-tile boards (phonies), for several lexica and both
      board sizes; leave values identical.
- [x] **2. Movegen + minimal protocol.** GADDAG generation of every move
      (tile plays, exchanges, pass) with scores; `gen` (debug list of all
      moves). Check: identical move lists and scores on the step 1
      positions (random games and random-tile boards, several lexica, both
      board sizes).
- [x] **3. Equity and static player.** Leave values, opening and endgame
      adjustments; best-play search with shadow pruning and WMP-based
      generation (these only pay off when looking for the best play, so
      they are checked here); static best move; `go static`.
      Check: identical equities and best moves.
- [x] **4. Game rules and autoplay.** Bag, racks, turn logic, paired bags;
      `autoplay` command. Check: identical games from the same seed (port
      `frand`'s ChaCha generator exactly). Challenge rules are left for
      step 12: autoplay and sims play with VOID.
- [x] **5. Montecarlo sim + BAI.** Sim, BAI, stopping conditions, endgame
      leaves; `go sim`, streaming `info`. Check: identical results with one
      thread and a fixed seed; statistical agreement with many threads.
      Plain sims (until stopped, or for a number of iterations or a time)
      and best-arm sims (the elite bot's rule) on any number of threads, as
      a std.Io group, best-arm results the same on any number; `go sim` in
      the background with progress, `stop` and `time`; endgame leaves (with
      two tiles or fewer in the bag a rollout's endgame is searched, each
      once a sim), the cross-check simming such positions too. Go's other
      stopping rules for `go sim` (classic, paired) are left out: the bots
      use best-arm, and the inference's mini-sim has the classic rule.
- [x] **6. Inference.** `rangefinder`; `go infer`. Check: identical with one
      thread and a fixed seed.
      The opponent's last move comes in the CGP's `lm` opcode (through
      tiles in parentheses, as open-protocols now specifies); the engine
      takes it back to get the position before it. Ported: the elite bot's
      path in budget mode (the deterministic one): mini-sims under the
      classic Stop99 rule (ported for them only), exhaustive enumeration
      of small leave spaces, prior sampling, the imputed complete
      posterior (marginal lifts to order 3, the value-first term for
      six-tile leaves, cross-fitted calibration), refinement rounds, and
      exchanges read by leave value. Leaves are dealt in Go's single-thread
      order and their mini-sims run in parallel, so the read is Go's
      one-thread read on any number of threads. `go sim ... infer <budget>`
      deals the opponent's racks from the posterior (best-arm and plain
      sims). Not ported: time-limited reads (not reproducible in Go
      either), the exchange sampler modes, order-4 marginals and other
      tuning knobs the bot leaves at their defaults, and two-turn (carry)
      reads.
- [x] **7. Endgame.** Zobrist hashing, negamax, TT, ABDADA, PV split;
      `go endgame`. Of the parallel searches only PV split and ABDADA are
      ported; the others in `endgame/negamax` are slower. (The Go Zobrist keys come from an unseeded RNG, so
      hashes are compared by property, e.g. incremental = from scratch.)
      Check: identical values (not PVs; ties can differ) on the endgame
      test suite.
      Done: the single-threaded search (`go endgame`), with Go's keys handed
      over (`zobrist-file`) and the same table size identical to Go's in
      value, best move and positions searched (`TestEndgameCrossCheck`),
      incremental move lists, keep-one and out-reply bounds and the pruned
      word graph included. From MAGPIE, by default: its move ordering
      (Go's magpieOrdering) with its lazy selection and stable root
      sort, aspiration windows, first replies one ply above the leaves,
      capped leaves in the first depth of a deeper search, both sides'
      root lists, playouts whose first move comes from the move lists,
      and its cached best-play search; the board keeps its squares as
      MAGPIE's does. The defaults search MAGPIE's tree and solve at its
      speed with its values. On several threads (the `threads` option),
      PV split by default, as Go's, or MAGPIE's ABDADA (`abdada`: each
      thread deepens on its own, the helpers starting deeper, and a
      younger move whose position another thread is in waits for a
      second pass); the threads keep their caches between solves
      (Workspace). The cross-check runs
      Go's ordering and MAGPIE's node for node, and the defaults by
      value, on one thread and on four with each parallel search.
- [ ] **8. Preendgame.** `go preendgame`. Check: identical results.
- [ ] **9. Best-move policy.** `go auto` (the elite bot's choice of
      algorithm). Check: identical choices.
- [ ] **10. WordSmog.** Alpha DAWG, alpha cross-sets, alpha movegen.
- [ ] **11. WASM build.** `wasm32` with shared memory and a Web Worker pool
      (or the `wasm32-emscripten` target with pthreads; decide at this step),
      postMessage adapter, a demo page. Needs COOP/COEP headers on Woogles.
- [ ] **12. Go side switch-over.** Shell setting `engine zig|go`; the Go
      engine stays available until the Zig core has covered a release.
      Challenge rules (single, double, void, five-point) come here, with
      games played through the interface.
- [ ] **13. Drop the Go-exact code.** Once the Go engine is no longer the
      reference: the engine uses Zig's own sort, `std.Random` (ChaCha) and
      `std.math` instead of `src/gocompat/`, which goes with its golden
      files and `tools/gogolden`. Seeded results change, so the tests that
      pin them are re-baselined in the same step against the engine's own
      results, and the Go cross-checks are retired.
- [ ] **14. ML.** Feature extraction, `Evaluator` interface, `ort` and
      `remote` backends.

## Testing

- **Port the Go tests** for each module, including their fixtures in
  `testdata`.
- **Cross-check harness** (Go, in `zigcore/crosscheck`): generates random
  positions with the Go engine, sends them to the Zig binary over UCGI and
  compares results. Runs in CI on a fixed number of positions and can be run
  longer locally.
- **Every way to reach a board:** a board the engine reaches without
  loading it (moves played on with incremental cross-set updates, a move
  played and taken back, a position loaded with its last move and taken
  back, as inference does) must give the same squares, moves, scores and
  best move as Go's board loaded from scratch (`TestBoardPathsCrossCheck`,
  `-crosscheck.pathgames`, `-crosscheck.takebacks`, using the debug
  commands `play`, `takeback` and `cgp`). Any new way of changing a board
  joins this test.
- **Memory:** unit tests allocate through `std.testing.allocator`, which
  fails a test that leaks or frees twice. Debug and ReleaseSafe builds of
  the engine report leaks at exit, and the cross-check harness fails a test
  whose engine reports one (CI runs it on a ReleaseSafe build). Long
  searches (sims, endgames) also get a soak test that watches the
  process's memory over many runs (`TestSoak`, `-crosscheck.soak <n>`:
  resident anonymous memory must stop growing after warm-up; mapped files
  don't count).
- **Determinism first:** compare single-threaded, fixed-seed results exactly.
  Multithreaded results are compared statistically.
- **Benchmarks:** keep a `zig build bench` with movegen-per-second,
  sim-iterations-per-second and endgame nodes-per-second, and record numbers
  next to Go and MAGPIE on the same machine. For the best-move search, the
  positions come from `TestWritePositions` in the cross-check package, Go
  times them with `BenchmarkStatic` and MAGPIE with
  `bench/magpie_bench.sh`; `zig build bench` can also break the search into
  phases (`BENCH_MODE=phases`) and sample a profile
  (`BENCH_PROFILE=<file>`, read with `llvm-symbolizer --inlining`). Sims
  are timed on the same kind of file (midgame positions): `BENCH_MODE=sim`
  (the second argument is the iterations; BENCH_PLIES, BENCH_THREADS,
  BENCH_BAI, BENCH_RIT), Go's `BenchmarkSim` and `bench/magpie_sim.sh`
  (a driver timing MAGPIE's own commands by wall clock: the sim time
  MAGPIE's CLI prints leaves part of the work out).
  `zig build test -Dtsan=true` runs the tests under ThreadSanitizer.

## Zig version policy

- Pin one Zig release (`minimum_zig_version` in `build.zig.zon`, exact
  version in CI). Start on 0.17.x.
- Upgrades are their own PRs: read the release notes, update the adapters
  and `build.zig`, and keep the full test suite and cross-checks passing.
  No feature work in an upgrade PR.

## Borrowing from MAGPIE

MAGPIE (jvc56/MAGPIE, C, GPL) and Macondo share data formats and ideas.
Improvements there can be ported here, keeping attribution in comments. The
Zig core is GPL-3.0 like the rest of Macondo; the WASM build shipped on
Woogles needs its source available.

## Open questions

- WASM threading: the core takes its concurrency from a `std.Io` passed in
  (worker groups, mutexes, events), so the WASM build supplies one: Zig's
  `Io.Threaded` on wasm32 with atomics and shared memory, or an `Io` of our
  own over a Web Worker pool. Which, and on which target, is decided at
  step 11.
- `info` payloads: plain tokens, or JSON for the richer sim and inference
  output? (Leaning: tokens, with JSON allowed in specific `info` fields.)
- Should weaker bot levels (`ai/bot/filters.go`) move into the core so bots
  can run in the browser?
- Does the socket bridge belong in the Zig binary (`--listen`) or in a small
  separate proxy?
