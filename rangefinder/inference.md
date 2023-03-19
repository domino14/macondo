Rough inference algorithm:

The last play made kept a certain set of tiles. Let's say the play used {X} tiles, keeping {Y} tiles.

We wish to know what tiles show up in {Y} most frequently (and least frequently).

We can pretend to be our opponent at the time they made the play. We iteratively set their rack to {X} + {R}, where {R} are random tiles from the bag, enough to fill their rack. Note that even though our rack is unseen to them, {R} cannot include our rack, since we know they don't have those letters.

We then generate moves statically. If their move is in the top Z moves (or we can set an equity cutoff), then {R} is a good estimate for what they kept.

We save {R} in memory, and generate a few thousand of these. 

Then, when we do sims, we can set their partial rack to each {R}, iteratively as well. Once we run out of {R} racks we can probably stop the sim. If we don't have enough {R} racks then it might mean their rack was just not easy to estimate, and we can just choose random racks. Another alternative is just to cycle through the different {R}s over and over again, drawing different tiles for the remainder of the racks.

We can also figure out the distribution of {R}s relative to what is left in the bag. If the different racks have many more Ss than would be expected by chance, then we can show this to the user. If they have fewer Js than would be expected by chance, we can show this to the user, etc.