# Pre-endgame

- [Back to Manual](/macondo/manual)
- [Back to Main Page](/macondo)

## What is it?

Macondo can solve an exhaustive pre-endgame with up to 6 tiles in the bag. The caveat
is that computers are not yet fast enough to fully take advantage of this (and
my algorithm is also not the most optimized).

Pre-endgames have always worked for bag-emptying plays since their initial
release, but the bug for non-bag-emptying plays has recently been fixed. Let's look
at a couple of example positions that illustrate how everything works. It's recommended to
read through this if you want to understand how Macondo's implementation of the pre-endgame algorithm works and what kind of data you can obtain.

## Position #1

This interesting position happened in a game between expert players Josh Sokol and Noah Walton (with the NWL20 lexicon). Thanks to Francis Desjardins for bringing this position to my attention. It is Josh's turn - he holds `DEFNNPT` and the board looks like this:

```
   A B C D E F G H I J K L M N O     ->                 Josh  DEFNNPT  394
   ------------------------------                       Noah           365
 1|=     '       =       ' D   = |
 2|  U       "       "     O -   |   Bag + unseen: (11)
 3|  p -       '   '       L     |
 4|' R   C       K A N J I S   ' |   A A E G I I I R S T U
 5|  I   O -     A     U         |
 6|  G   T   "   I   " I     "   |
 7|  H ' E     ' Z '   C   L O O |
 8|= T E D       E   B Y W O R D |
 9|    Q       ' N '     A X E   |
10|  R u B I G O S   "   I   "   |
11|F   A   -       W E A V E     |
12|O   T -       '       E     ' |
13|V   E       '   L O U R Y     |
14|E N S N A R L     H M     -   |
15|A     '       T E M P '     = |
   ------------------------------
```

Josh is up 29 points and there are 4 tiles in the bag (11 unseen minus Noah's 7).
Load it into Macondo:

```
load cgp 12D2/1U10O2/1p10L2/1R1C3KANJIS2/1I1O3A2U4/1G1T3I2I4/1H1E3Z2C1LOO/1TED3E1BYWORD/2Q4N3AXE1/1RuBIGOS3I3/F1A5WEAVE2/O1T8E3/V1E5LOURY2/ENSNARL2HM4/A6TEMP4 DEFNNPT/ 394/365 0 lex NWL20;
```

### Scoping analysis to one move with `-only-solve`

Running a full `peg` on 4-in-the-bag with non-bag-emptying candidates is slow.
If you already have a specific candidate in mind — Josh finds `2L P(O)ND` and wants
to know whether it is a guaranteed winner before committing — you can scope the
solve to just that move:

```
peg -only-solve "2L P.ND" -endgameplies 4
```

`-only-solve` accepts a move in the same notation as `gen` and `commit` (the `.`
stands for the board tile already at O2). You can repeat the flag to compare a
short list:

```
peg -only-solve "2L P.ND" -only-solve "2L F.ND" -endgameplies 4
```

`-endgameplies 4` is enough for this position (and is the default). If you need
to resolve close situations involving tile-stick endgames, for example, bump it to 6 — but expect much longer runtimes.

### Reading the PEG output

After the solve completes you will see something like:

```
Winner:  2L P.ND
Play                Wins    %Win    Spread   Outcomes
 2L P(O)ND          7884.0  99.55            👍: [AAE]G [AAE]I [AAE]R ... 👎: [III]A [III]E [III]G [III]S [III]T
```

**Columns:**

- **Wins** — how many of the possible bag-draw permutations end in a win for
  Josh, counting only permutations where the win is *guaranteed* (see below).
- **%Win** — Wins / total permutations × 100.
- **Spread** — average score difference over winning permutations (used as a
  tiebreaker when two plays have the same win count).
- **Outcomes** — a compact list of every draw permutation that was fully solved.
  `👍` = win, `👎` = loss, `🤝` = tie.

**Outcome notation:** `[ABC]D` means Josh drew tiles A, B, and C, and D is the tile left in the bag for Noah. The brackets around ABC tell us that the order of those tiles does not matter.
For a 4-in-bag position where Josh's play uses 4 tiles off his rack, he draws 4 and leaves 0 in the bag —
so the outcomes list will look like `[ABCD]` with nothing after the bracket.

### Stepping into one eventuality with `-tileorder`

The outcome list for 2L P(O)ND contains mostly wins, but `[III]A` appears in the
`👎` column.

To investigate, replay that exact draw:

```
commit 2L P.ND -tileorder IIIA
```

`-tileorder` takes the bag draw in left-to-right = first-drawn order, the same
convention as the outcome notation. `IIIA` means Josh draws I, I, I and A stays in
the bag. Macondo sets Josh's rack to his new tiles, auto-assigns Noah the remaining
unseen tiles (anything not in the tileorder), and makes the move.

After committing, the board looks like this — it is now Noah's turn:

```
   A B C D E F G H I J K L M N O                        Josh           408
   ------------------------------    ->                 Noah  AEGRSTU  365
 1|=     '       =       ' D   = |
 2|  U       "       "   P O N D |   Bag + unseen: (8)
 3|  p -       '   '       L     |
 4|' R   C       K A N J I S   ' |   A E F I I I N T
 5|  I   O -     A     U         |
 6|  G   T   "   I   " I     "   |
 7|  H ' E     ' Z '   C   L O O |
 8|= T E D       E   B Y W O R D |
 9|    Q       ' N '     A X E   |
10|  R u B I G O S   "   I   "   |
11|F   A   -       W E A V E     |
12|O   T -       '       E     ' |
13|V   E       '   L O U R Y     |
14|E N S N A R L     H M     -   |
15|A     '       T E M P '     = |
   ------------------------------
```

Noah has `AEGRSTU` and there is 1 tile left in the bag (the A).
Josh is up 43 points heading into a 1-in-the-bag pre-endgame.

### Noah's 1-in-bag PEG

From Noah's perspective (run `peg` from the current game state with Noah on turn):

```
peg -endgameplies 4
```

```
Winner:  2A A.GUSTER
Play                Wins    %Win    Spread   Outcomes
 2A A(U)GUSTER      2.0     25.00   0.12     👍: [F] [T] 👎: [A] [E] [I] [N]
15H (TEMP)TER       2.0     25.00   -16.38   👍: [E] [F] 👎: [A] [I] [N] [T]
15H (TEMP)URA       1.0     12.50            👍: [A] 👎: [E] [F] [I] [N] [T]
```

(All other plays lose every endgame.)

Noah's two best plays — `A(U)GUSTER` and `(TEMP)TER` — each win exactly 2 of the
8 possible bag draws. Crucially, **neither of them wins when A is the bag tile**.
`(TEMP)URA` is the only Noah play that wins the A-in-bag endgame, and it wins just
that one eventuality out of eight (12.5 %).

Noah does not know what is in the bag. From his perspective TEMPURA looks like an
inferior play: it has fewer wins and worse expected spread than his top options.
A rational opponent will not choose TEMPURA.

### Why `[III]A` is marked a loss — guaranteed wins vs "both sides solve PEG"

This is the conceptual heart of Macondo's pre-endgame model.

#### Guaranteed wins mode (current)

Macondo marks an eventuality as a **win** only if Josh wins *regardless of which
play Noah makes*. For `[III]A`, Josh will not win if Noah plays TEMPURA.

Result: `[III]A` is `👎` — not a guaranteed win for Josh, even though it would not be rational for Noah to play TEMPURA.

#### Full PEG on both sides (hypothetical — not yet implemented)

Imagine a mode where both players solve their sub-PEG optimally. Noah would look at
his 1-in-bag position, compute his best EV plays, and pick among them — specifically
AUGUSTER or TEMPTER, both of which **lose** when A is the bag tile. Under this
model, Josh *does* win the `[III]A` eventuality, because Noah's objectively optimal
responses all lose it.

In this mode `[III]A` would be a `👍` for Josh's 2L P(O)ND.

"Both sides solve PEG" would be the right model for
computing true game-theoretic value, but it requires solving the nested PEG on the
opponent's turn (expensive and not yet implemented).

### The full result

Going back to the beginning, we can solve for the play 2L P(O)ND and obtain this result. Running at 6 endgame plies (note: ~11 minutes at 4 plies, ~3 hours at 6 plies on a
modern machine):

```
Winner:  2L P.ND
Play                Wins    %Win    Spread   Outcomes
 2L P(O)ND          7884.0  99.55            👍: [AAE]G ... 👎: [III]A [III]E [III]G [III]S [III]T
```

2L P(O)ND wins 7884 of 7920 possible draw permutations (99.55 %). The five losing
outcomes are `[III]A`, `[III]E`, `[III]G`, `[III]S`, and `[III]T` — you can go through these one by one using the above procedures. In each of those cases there exists some Noah play that can beat Josh, even if sometimes it is not Noah's globally optimal pre-endgame play.

Still, if you were Josh at the board, 2L P(O)ND is overwhelmingly correct: it is a
guaranteed win in 99.55 % of all possible bag draws.

#### What about the other plays?

It is prohibitive to examine many shorter plays fully. Look at the writeup on shortcuts below for reasons why and for various things you can try.

However, `peg` by default tries to have sensible shortcuts. One of the ones you should use often is `-early-cutoff true`. This stops analyzing pre-endgame lines that can't do better than the best one you've found so far.

For this particular 4-in-the-bag preendgame, if you run `peg -early-cutoff true` this will automatically examine all plays that _empty the bag or leave at most 1 tile in the bag_, cutting off plays that can't win. Use this if you want to find the winningest play for which this condition is true.

You will need to set `-max-tiles-left -1` in order to examine all plays. This will take an exceedingly long time.

The following was computed with `max-tiles-left 0`, so it only shows bag-emptying plays. After some time (about 12 minutes on my computer, using 4 endgame plies by default), we get the following moves:

```
Winner:  I1 DEF.T
Play                Wins    %Win    Spread   Outcomes
 I1 DEF(A)T         7872.0  99.39            👍: [AAEG] [AAEI] ... 👎: [IIIT] [IIIU]
 E9 P(I)NET(A)      7692.0  97.12            👍: [AAEG] [AAEI] ... 🤝: [IIIU] 👎: [AIII] [AIIT] [IIIT]
 I1 PED(A)NT        7692.0  97.12            👍: [AAEG] [AAEI] ... 🤝: [IIIU] 👎: [AIII] [AIIT] [IIIT]
...
```

Note that I1 DEF(A)T only loses with [IIIT] and [IIIU] draws.

But 2L P(O)ND is slightly winninger at 99.55%! But the solver doesn't find it by default because it doesn't empty the bag. The following section explains how to calculate the math for this.

#### Calculating number of wins and permutations

The `Wins` column counts **ordered bag-draw permutations**, not unordered
combinations. For a 4-in-the-bag position with 11 unseen tiles, the total is
P(11, 4) = 11 × 10 × 9 × 8 = **7920**. Duplicate tiles (the two A's and three
I's in this position) are treated as distinguishable when counting — that's
how the denominator lands on 7920.

Each label in the `Outcomes` column covers a group of those 7920 permutations.
**The bracket position controls how many permutations each label is worth.**

##### Bag-emptying plays — `[ABCD]` (all four tiles inside the brackets)

Josh plays 4 tiles, draws all 4 out of the bag, and the bag is empty. Every
ordering of the same four tiles ends him in the same position, so the label
groups all 4! orderings:

```
perms per [ABCD] label = 4! = 24
```

So for `I1 DEF(A)T`:

- `[IIIT]`: 24 permutations.
- `[IIIU]`: 24 permutations.
- **Losses: 24 + 24 = 48. Wins: 7920 − 48 = 7872 (99.39%).** ✓

##### 3-tile plays — `[ABC]D` (three inside, one outside)

Josh draws only 3 tiles; the 4th tile stays in the bag and Noah draws it on
his turn. The label now pins the outside tile to the 4th position in the
bag's draw order. Its weight is:

```
perms per [ABC]D label = (orderings of Josh's 3 tiles) × (copies of D in the unseen pool)
```

For `2L P(O)ND`, Josh's 3 tiles are always `{I, I, I}`, so the orderings term
is 3! = 6. The multiplier depends on how many copies of the leftover tile are
in the unseen pool `A A E G I I I R S T U`:

| Label     | 3-tile orderings | Copies of leftover  | Permutations |
|-----------|------------------|---------------------|--------------|
| `[III]A`  | 3! = 6           | 2 (two A's unseen)  | **12**       |
| `[III]E`  | 6                | 1                   | 6            |
| `[III]G`  | 6                | 1                   | 6            |
| `[III]S`  | 6                | 1                   | 6            |
| `[III]T`  | 6                | 1                   | 6            |

**Losses: 12 + 6 + 6 + 6 + 6 = 36. Wins: 7920 − 36 = 7884 (99.55%).** ✓

##### Why five losing labels beat two

Each bag-emptying `[XYZW]` label is worth 24 permutations, but each 3-tile
`[XYZ]W` label is worth only 6–12. So `2L P(O)ND`'s five loss labels
(36 permutations) represent fewer actual losses than `I1 DEF(A)T`'s two loss
labels (48 permutations), and that's where the extra 12 wins come from.

Conceptually: evaluating `2L P(O)ND` as a 3-tile play resolves each losing
multiset into a finer-grained question — "does it also lose when a *specific*
tile is the one left in the bag?" In tempura, when the bag contains
`{I, I, I, T}` and Josh empties it, he loses; but when Josh plays `2L P(O)ND`
and leaves one of those tiles behind, he only actually loses if T is the one
left (Noah getting T after the play still beats Josh). The bag-emptying
view counts the entire multiset `[IIIT]` as a loss; the 3-tile view separates
it and only counts the cases that truly lose.

*(If you want to verify the weighting in code, see `generatePermutations` and
`product` in `preendgame/peg_generic.go`. `product` multiplies "tiles of this
type still available" as it assigns each draw slot — that gives each distinct
tile-type sequence its weight, which the solver sums inside each outcome label.)*

### The plot thickens

There are two moves that guarantee that Josh actually never loses!

You can play `E5 FET`:

```
E5 FET             7914.0  99.92 👍: [AAE]G [AAE]I ... 🤝: [III]A
```

Or `I3 P(A)NT`:
```
I3 P(A)NT          7902.0  99.77            👍: [AAE]G [AAE]I ... 🤝: [AII]I
```

Note that both of these moves result in, at worst, a tie!

## How do nested pre-endgames work?

In the last game, we showed a position where Josh solves a specific situation for a 4-in-the-bag pre-endgame. Since the play only leaves 1 tile in the bag, by the time it is Josh's turn again, the bag will almost always be empty (unless Noah passes, which is a rare situation - that we can still handle, if needed).

A more interesting situation can be found below, which requires the player on turn to _again_ solve a pre-endgame with the bag still not empty.

### Position #2

This game was between Matthew Tunnicliffe and Jim Brennan with the CSW21 lexicon.

```
   A B C D E F G H I J K L M N O     ->  Matthew_Tunnicliffe  ?ANNOPY  344
   ------------------------------                Jim_Brennan           368
 1|B E D E L     =       '     = |
 2|R - R     "       "     U -   |   Bag + unseen: (10)
 3|O   I T   Q '   '     O M     |
 4|W   B I D I   '     Y U M   ' |   A E E E G I L N R S
 5|N     X I           A T       |
 6|E "     G "       T   R   "   |
 7|S   V O I L E   ' O K A '     |
 8|T     ' T   D I S P A C E D = |
 9|    '       '   ' A W E ' O   |
10|  "       "       Z   s   F A |   Turn 25:
11|        -           -       R |   Jim_Brennan played 15A INRO for 25 pts fro
12|'     -       '       -   G O |   m a rack of INOR
13|    -       '   '       - A H |
14|  -   J U V I E   "     U T A |
15|I N R O       F L E N C H E S |
   ------------------------------
```

You can load it like:

```
load cgp BEDEL10/R1R9U2/O1IT1Q5OM2/W1BIDI4YUM2/N2XI5AT3/E3G4T1R3/S1VOILE2OKA3/T3T1DISPACED1/9AWE1O1/9Z1s1FA/14R/13GO/13AH/3JUVIE4UTA/INRO3FLENCHES ?ANNOPY/AEGILNS 344/368 0 lex CSW21;
```

Or, since the [game is found on cross-tables](https://www.cross-tables.com/annotated.php?u=39443%230%2523#26#), you can do:

```
set lexicon CSW21
load xt 39443
```

Here, Matthew played `13M P(AH)` for 32 points. We can examine this move by doing:

```
peg -only-solve "13M P.." -skip-deep-pass false
```

**(Note, this is slow! See below)**

We set `-skip-deep-pass` to false here to make sure that the algorithm examines passes. For this position and many others it likely doesn't really make a difference to examine passes though. They are quite rare in most pre-endgames.

By not specifying endgame plies it defaults to 4, which is sufficient for many endgames.

Note that you can just do `peg -max-tiles-left -1` as well to examine every move. But again, for non-bag-emptying moves, this is going to be very slow without a supercomputer. On my 16-core 12th gen Intel processor it takes around 2h48m to analyze _just this play_!

For many pre-endgames, you might be able to get away with just examining plays that empty the bag. It will be _much_ quicker than examining all plays. You can do this with `peg -max-tiles-left 0`.

#### Solution for P(AH)

PAH has a 27.92 guaranteed win%, counting ties as half a win.

```
Winner: 13M P..
Play                Wins    %Win    Spread   Outcomes
13M P(AH)           201.0   27.92            👍: AEI AEL AES AGI AGL AGS AIE AIG AIL AIN AIR AIS ALE ALG ALI ALN ALR ALS ANI ANL ANS ARI ARL ARS ASE ASG ASI ASL ASN ASR EIA EIG EIN GAI GAR GAS GEI GER GES GIA GIE GIN GIR GIS GLI GLR GLS GNI GNR GNS GRA GRE GRI GRN GRS GSA GSE GSI GSN GSR LAG LAI LAS LGA LGI LNI LRI LSI NAI NAR NGS NIA NIS NLS NRA NRS NSI NSR RIA RIL RIN RIS SAE SAG SAI SAL SAN SAR SEA SEE SEG SEI SEL SEN SER SGA SGE SGI SGL SGN SGR SIA SIE SIG SIL SIN SIR SLA SLE SLG SLI SLN SLR SNA SNE SNG SNI SNL SNR SRA SRE SRG SRI SRL SRN 🤝: LES LGS LIS LNS LRS NAS NEI NER NES NGI NGR NIE NIR NLI NLR NRE NRI NSA NSE RIG 👎: AEE AEG AEN AER AGE AGN AGR ANE ANG ANR ARE ARG ARN EAE EAG EAI EAL EAN EAR EAS EEA EEE EEG EEI EEL EEN EER EES EGA EGE EGI EGL EGN EGR EGS EIE EIL EIR EIS ELA ELE ELG ELI ELN ELR ELS ENA ENE ENG ENI ENL ENR ENS ERA ERE ERG ERI ERL ERN ERS ESA ESE ESG ESI ESL ESN ESR GAE GAL GAN GEA GEE GEL GEN GIL GLA GLE GLN GNA GNE GNL GRL GSL IAE IAG IAL IAN IAR IAS IEA IEE IEG IEL IEN IER IES IGA IGE IGL IGN IGR IGS ILA ILE ILG ILN ILR ILS INA INE ING INL INR INS IRA IRE IRG IRL IRN IRS ISA ISE ISG ISL ISN ISR LAE LAN LAR LEA LEE LEG LEI LEN LER LGE LGN LGR LIA LIE LIG LIN LIR LNA LNE LNG LNR LRA LRE LRG LRN LSA LSE LSG LSN LSR NAE NAG NAL NEA NEE NEG NEL NGA NGE NGL NIG NIL NLA NLE NLG NRG NRL NSG NSL RAE RAG RAI RAL RAN RAS REA REE REG REI REL REN RES RGA RGE RGI RGL RGN RGS RIE RLA RLE RLG RLI RLN RLS RNA RNE RNG RNI RNL RNS RSA RSE RSG RSI RSL RSN
```

This solution involves solving 9,018,256 endgames!

#### Examining a specific permutation: GRL

Let's think about the following situation: G, R, and L are in the bag, in that order, which means that Jim's rack must be AEEEINS.

Matthew plays P(AH) like above, draws the G, and then Jim has a few hundred choices. Macondo on iterates through all of these choices and tries them, _without actually solving the pre-endgame on Jim's end_. Note that in the discussion above we mentioned that the pre-endgame is not actually solved on both ends.

Let's say that Jim plays `K7 (KAW)A` for 22 points. He would then draw the R.

Now, it's Matthew's turn again and his rack is `?AGNNOY`. Unseen to him is `EEEILNRS`. Matthew doesn't know that the L is in the bag. He can now solve the 1-in-the-bag preendgame (1-PEG) - and indeed, that is what Macondo does.

Note that this differs from the first game. We had no nested pre-endgame in the first game; the opposing side did not actually solve the PEG from their perspective. We only solve the nested pre-endgame for the side-to-solve and not both sides.

##### Nested pre-endgame solution

The solution for this 1-PEG is:

```
Winner:  1H ANONYm
Play                Wins    %Win    Spread   Outcomes
 1H ANONYm          7.0     87.50            👍: [E] [I] [N] [R] [S] 👎: [L]
10F NYAN(ZAs)       ---     ---              👍: [E] [L] [N] 🤝: [I] 👎: [R]    ❌
 N2 GApY            5.0     62.50            👍: [E] [I] [S] 👎: [L] [N] [R]
13G OrGAN           5.0     62.50            👍: [E] [I] [S] 👎: [L] [N] [R]
...
❌ marks plays cut off early
```

Macondo's PEG uses "guaranteed wins" mode, which means we assume that the side-to-solve would always find their "best" move. We define "best" here as most points, where a win scores 1 point and a tie scores 0.5 points.

In the case above, we assume that Matthew would find ANONYm and play it, as it wins the most games (7 out of 8). He doesn't know that the L happens to be in the bag, for this eventuality. Unfortunately, it is, and he would lose in this case.

Therefore, for the case that GRL is in the bag, in that order, the move P(AH) cannot be called a win. Even if KAWA happens to be the only move the opponent could make that could result in a loss for Matthew, Macondo is pessimistic and will call P(AH) a loss. Note that we don't have to find any other moves besides KAWA. KAWA is enough to call P(AH) a loss in this case.

##### Showing specific result for this permutation

If you run `peg -only-solve "13M P.." -skip-deep-pass false -eventuality GRL` it will
solve this particular play for only this eventuality. Note: this solve will be single-threaded, so it might not be as fast as you'd expect. On my computer this takes a
bit over a minute:


```
═══════════════════════════════════════════════════
 Eventuality verdict: LOSS for 13M P..
═══════════════════════════════════════════════════

 We play  13M P.. +32
   Rack before: ?ANNOPY   Bag before: [LRG]   Opp: AEEEINS
   Rack after:  ?AGNNOY   Bag after:  [LR]

 Opp replies checked: 8
 Decisive reply: 10J .A. +22
   Opp rack after: EEEINRS   Bag after: [L]

 ── Nested pre-endgame (depth 1) ──
   Our rack: ?AGNNOY   Opp: EEEINRS   Bag: [L]
   Unseen pool: E×3 I L N R S
   Best play under uncertainty:  1H ANONYm +43
     E ×3       → WIN
     L          → LOSS   ← actual bag content
     N          → WIN
     R          → WIN
     I          → WIN
     S          → WIN
   Verdict at depth 1: LOSS

═══════════════════════════════════════════════════
```

This shows you exactly why this was labeled a loss.

##### What if more than one play is tied for first in the nested pre-endgame?

Let's imagine for the sake of argument that there was another play that also won 7/8 endgames in the nested PEG, but loses when the opponent draws an R. So Matthew would have a choice between two 7/8 plays, only one of which loses when he draws the L in the bag. But we can't assume he'll choose the correct one here. Macondo will still be pessimistic and call this a potential loss.

## Shortcuts and when to disable them

Three on-by-default settings make nested PEG practical on real positions.

### `-max-nested-depth 1` (default)

When a non-emptying line goes deep, the recursion proceeds as follows:

1. Josh plays non-emptying P1 (outer PEG).
2. Opponent's replies are enumerated (a flat list, not a sub-PEG).
3. Opponent plays non-emptying P2 → Josh's turn, bag still non-empty.
4. **Nested sub-PEG for Josh** (`nestedDepth=1`): generates all of Josh's candidate
   replies × all remaining sub-bag-permutations and evaluates the resulting positions.
5. Inside that nested PEG, if Josh plays another non-emptying move and the opponent
   again replies non-emptying, Josh's turn would occur a third time.
6. That third turn would require `nestedDepth=2` — which the cap blocks.

The default cap of `1` therefore means: Josh gets **one** nested sub-PEG (step 4).
If within that sub-PEG further non-emptying exchanges would require a second nested
sub-PEG, those continuations are skipped. This is an approximation: a skipped
continuation might have been a loss for Josh, so the win count can be slightly
optimistic compared to the unlimited solve.

Without any cap, this particular position produced 4.6 million nested calls and 518
million endgame solves in three hours with only 29 candidate plays processed.

To disable the cap and run the full exhaustive solve:

```
peg -max-nested-depth -1 -maxtime 3600
```

Expect runtimes measured in days or weeks on positions with many tiles in the bag.
Use `-max-nested-depth 0` to skip all non-bag-emptying analysis entirely (equivalent
to `-max-tiles-left 0`).

### `-skip-deep-pass true` (default)

Passes are legal pre-endgame moves — sometimes the right call. But in the nested
sub-PEG, a pass generates another candidate that can itself trigger further nesting.
Repeated pass chains in nested sub-games are vanishingly rare in real positions, yet
they multiply the search space.

With this option on, passes are suppressed in the nested sub-PEG candidate
generation *except* when the previous move was also a pass — so the bounded
"pass → pass → game ends" branch is always evaluated correctly. Pass as one of our
top-level PEG candidates is always considered regardless of this setting.

To restore full pass generation in nested sub-games:

```
peg -skip-deep-pass false
```

### `-max-tiles-left 1` (default)

By default, Macondo only analyzes plays that leave at most 1 tile in the bag
after the play. This scales with bag size: at 4-in-bag, plays using 3 or 4 tiles
are analyzed (leaving 1 or 0); passes and short plays (leaving 2+) are skipped.
At 5-in-bag, plays using 4 or 5 tiles are analyzed, and so on.

The combinatorial explosion from non-bag-emptying plays grows with the number
of tiles remaining in the bag, so skipping plays that leave many tiles behind
keeps the default run fast while still analyzing the most practically relevant
candidates — plays that nearly or completely empty the bag.

To analyze all plays regardless of tiles left (full exhaustive solve):

```
peg -max-tiles-left -1 -maxtime 3600
```

To only analyze plays that completely empty the bag:

```
peg -max-tiles-left 0
```

Note: `-only-solve` always overrides this setting. If you specify a particular
move with `-only-solve`, it will be solved even if it leaves many tiles in the bag.

### Clarity on output

The solve-start log (`preendgame-solve-called`) records `nested-depth-limit`,
`skip-deep-pass`, and `max-tiles-left` so you know which approximations are in
effect. The `peg-status` ticker (printed every 60 seconds) shows live performance
counters.

## Further options

`peg` has many more options — thread count, time limits, skipping non-emptying
plays, providing partial opponent rack info, and more.
`commit` supports `-tileorder` for any position where you want to force a specific
draw order. Run `help peg` and `help commit` inside the Macondo shell for the full
option reference.

