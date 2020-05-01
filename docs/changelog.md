# v0.4.2

Bug fix and enhancement release

- Changed the semantics of `add` by only allowing it to add on to a list of plays, but not commit the play
- Added a `commit` command, which does what `add` used to do
- Refactored game history with protobuf to fix various bugs and hacks. It is much cleaner now.
- Add a `sim continue` command to allow a sim to continue from where it left off
- Add a `sim trim` command to delete bottom performers of a simulation.
- Add an `export` command to allow saving to a .gcg file
- `add` and `commit` commands are less picky about including the "through" letters in the play, as opposed to requiring a `.` character for these letters
- Add a `setlex` command to set the lexicon of a game after loading it
- Add color board display for terminals that support it
- Fix bug where character encoding for GCGs containing CRLF characters was misparsed
- Various other usability fixes

# v0.4.1

Bug fix release

- Fixed shell `sim` command so we can do 3-ply and above
- Display endgame results in a nicer format
- Allow `add exchange ABC` and `add pass` commands to exchange and pass
  without move generation
- Default to info-level logging (turn on debug with DEBUG=1 environment variable)
- Keep text on board within a small horizontal range for smaller windows

# v0.4.0

The first pre-alpha version.

- Added Monte Carlo simming
- Added `autoplay` to the shell interface
