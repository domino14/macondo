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
