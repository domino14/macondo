# AI Explainability

- [Back to Manual](/macondo/manual)
- [Back to Main Page](/macondo)

## What is it?

Macondo has an experimental feature to allow you to explain a move with generative AI. If you are curious, the prompts used are in the `/explainer` directory.

See this position:

```
   A B C D E F G H I J K L M N O     ->              player1  ACDEPQU  400
   ------------------------------                    player2           440
 1|C O X A     B =   R O P I N G |
 2|  -   T   D I N G Y       -   |   Bag + unseen: (16)
 3|    - t     Z O ' A I N E E   |
 4|'     A       V     F O W T H |   E E G I I I L L N O O R R S S U
 5|      S -     A     F O E   U |
 6|  L E K E "   T   M Y     " T |
 7|    R E L A T I V E     '   I |
 8|A R E D       N       '   B A |
 9|    '       ' g '       ' R   |
10|  "       "       "     W O   |   Turn 0:
11|        -           -   H A   |
12|'     -       '       - I D ' |
13|    -       '   '       M E   |
14|  -       "       "       N   |
15|=     '       =       J U S T |
   ------------------------------
```

You can load it as follows:

```
macondo> load cgp COXA2B2ROPING/3T1DINGY5/3t2ZO1AINEE1/3A3V2FOWTH/3S3A2FOE1U/1LEKE2T1MY3T/2RELATIVE4I/ARED3N5BA/7g5R1/12WO1/12HA1/12ID1/12ME1/13N1/11JUST ACDEPQU/ 400/440 0 lex CSW24;
```

### Running AI Explainability

To test it out, please create a Gemini API key here:

https://ai.google.dev/gemini-api/docs

You can click Get a Gemini API Key in that page, and place it in an environment variable `GEMINI_API_KEY`.

Then, type the following command into Macondo:

```
macondo> script scripts/lua/genai_explain.lua
```

#### Gemini 2.5 Pro Experimental respnse:

This should print something like the following after a few seconds:

Model response: Okay, let's break down this position. The simulation identifies `12K QU(ID)` as the strongest play. Here's why:

*   **Excellent Future Potential:** Although this play doesn't score the most points this turn (28 points vs. 30 for `12J EQU(ID)`), it sets you up much better for your *next* turn. Your average score after playing `12K QU(ID)` is projected to be 44.5 points, significantly higher than any other option (which range from 32 to 35 points). This is largely due to the excellent leave.
*   **Strong Leave Quality:** Playing `QU(ID)` leaves you with ACDEP. This gets rid of the restrictive Q and U tiles while keeping a balanced mix of vowels (A, E) and useful consonants (C, D, P). Compared to leaves like DEQU or ACQU from other considered plays, ACDEP offers much more flexibility for future scoring and bingo opportunities. The bag is balanced (8 vowels, 8 consonants), so this slightly consonant-heavy leave (2V/3C) should draw well.
*   **Setup Opportunity:** The detailed stats show a specific high-scoring follow-up: `15G PREAD(JUST)`. This play scores 57 points and has a solid 11.5% chance of being available next turn if you draw an R. This possibility contributes significantly to the high average score projected for your next turn.
*   **Winning Chances:** These factors combine to give `12K QU(ID)` the highest win percentage by a considerable margin (37.8% vs. 24-30% for the others). While you're currently 40 points behind, this play maximizes your chances to score well next turn and overtake your opponent, without drastically increasing your opponent's immediate scoring threats (their average score and bingo % are standard).

### Models

At the moment of writing this (April 14, 2025) the model we are using, Gemini 2.5 Pro, is perhaps the strongest AI model out there. An experimental version of it is available for free. You should be able to run around 25 explanations per day. Normally, an explanation with this model would cost around $0.04.

You can check the Lua script above (`in the scripts/lua/genai_explain.lua`) to update the model. The default value is `gemini-2.5-pro-exp-03-25`. When the model becomes generally available, this default value is likely to change. You can also change the model used by modifying the `GEMINI_MODEL` API key. For example, `gemini-2.0-flash` is 1-2 orders of magnitude cheaper, and the quality of the response is almost as good:

#### Gemini 2.0 Flash response:

Model response: In this position, `12K QU(ID)` performed best. This is why:

*   The play `12K QU(ID)` has a significantly higher win % than the other plays, leading the closest play, `5D (S)CAP(A)`, by over 7%. We are behind by 40 points, and at this stage in the game (16 tiles left in the bag) it is more important to focus on improving our win % and less on equity (score + rack).

*   The play `12K QU(ID)` leaves the tiles ACDEP in the rack, which has a more balanced leave (3 consonants, 2 vowels). The fact that there are equal numbers of consonants and vowels left in the bag makes this balanced leave even more valuable.

*   There is a setup opportunity following `12K QU(ID)`. The play `15G PREAD(JUST)` has a 13.52% chance of being played next turn, given the right draw of R. This is a high-scoring play for 57 points, which could help close the 40-point gap.