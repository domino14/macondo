# v0.10.0 (November 22, 2024)
- Heat maps (in CLI) and in-depth sim stats for Monte Carlo
- Multiple variations support for endgame (see option `-multiple-vars`)
- Argument support for Lua scripts
- More correct statistical tests for Monte Carlo stopping condition
- Spanish rules and lexicon support
- CSW24 support
- Default lexicon set to NWL23

# v0.9.0 (April 3, 2024)
- Generic N-in-the-bag pre-endgame, with N up to 6
    - When a pre-endgame play shows up as a win for a given set of tiles in the bag, it is always a sure win in this case, provided we play our best endgame.
    - For bag-emptying plays, it is always correct.
    - For non-bag-emptying plays, it is almost always correct. A bug fix in the future will make it always correct.
        - The issue is if the bag is still not empty by the time it is our turn again. Macondo will not solve the inner/recursive pre-endgame exactly, but will instead optimize and find any win for ourselves. This is wrong.
    - Overall it is very slow for N > 1, especially if we're examining non-bag-emptying plays. See `help peg` for more options.
- Endgame speedups of 2x or more in some cases:
    - Endgame LazySMP implementation can now use 8-10 threads or so for best performance, due to aspiration search in multithreaded mode
    - Added PVS/Negascout for an extra speed boost to the endgame
- Many other bug fixes and improvements to memory usage, speed, etc.

# v0.8.0 (Jun 26, 2023)
- Made LazySMP implementation a bit better (can now use around 6 threads for best
endgame performance)
- Fully exhaustive 1-in-the-bag pre-endgame implementation.

# v0.7.0 (Jun 16, 2023)
- Many bug fixes from the last refactor
- Significant (can be 20X or more) speed increase in endgame engine for complex endgames.
- Allow endgames to be multi-threaded. (Use up to 3-4 threads. Any more makes it slower).

# v0.6.0 (Mar 19, 2023)
- Significant refactoring
- Support multi-letter tiles (for Catalan)
- Use KWG and KLV files instead of old gaddag/olv leave files
    - Requires use of wolges for now (https://github.com/andy-k/wolges) for kwg/klv file generation
- Speed up Monte Carlo sim by about 40%; make it zero-allocation

# v0.5.0 (Mar 8, 2023)
- Macondo is now in beta. It still is command-line only, however.
- Add an "infer" command that attempts to infer what your opponent kept
- Add inference-based Monte Carlo simming
- Add a new simming bot for autoplay and other uses

# v0.4.13 (Jan 28, 2023)
- Changes to puzzle maker for liwords; add a total equity loss limit
- Remove obsolete protobuf fields
- A few bug fixes

# v0.4.9 (Aug 20, 2022)

- Drastic optimizations to speed, allocation, and memory usage
- Experimental win% calculation

# v0.4.5 (rc)

- Add challenge rules and `challenge` command to the shell to handle these.
- Add end-of-game pass/challenge if challenge rule is not VOID.

# v0.4.4 (May 24, 2020)

- Add opp score to autoplay log
- Add bingo tracking in autoplay log and stats
- Create code for configurable pre-endgame heuristic values. Current heuristics are blank but can be iterated on.
- Create utility functions to return all words created by a play, and to validate a play's legality (not word legality but rule legality)
- Fix a few GCG import and export issues

# v0.4.3 (May 7, 2020)

This is a code organization release with no new features.

protobuf files have been moved around for ease for importing by other projects.

# v0.4.2 (May 6, 2020)

Bug fix and enhancement release

- Include NWL18 gaddag by permission
- Changed the semantics of `add` by only allowing it to add on to a list of plays, but not commit the play
- Added a `commit` command, which does what `add` used to do
- Refactored game history with protobuf to fix various bugs and hacks. It is much cleaner now.
- Added a lot of helpful options to `autoplay` command
- Add a `sim continue` command to allow a sim to continue from where it left off
- Add a `sim trim` command to delete bottom performers of a simulation.
- Add an `export` command to allow saving to a .gcg file
- `add` and `commit` commands are less picky about including the "through" letters in the play, as opposed to requiring a `.` character for these letters
- Add a `setlex` command to set the lexicon of a game after loading it
- Add color board display for terminals that support it
- Fix bug where character encoding for GCGs containing CRLF characters was misparsed
- Various other usability fixes

# v0.4.1 (Apr 9, 2020)

Bug fix release

- Fixed shell `sim` command so we can do 3-ply and above
- Display endgame results in a nicer format
- Allow `add exchange ABC` and `add pass` commands to exchange and pass
  without move generation
- Default to info-level logging (turn on debug with DEBUG=1 environment variable)
- Keep text on board within a small horizontal range for smaller windows

# v0.4.0 (Apr 7, 2020)

The first pre-alpha version.

- Added Monte Carlo simming
- Added `autoplay` to the shell interface
