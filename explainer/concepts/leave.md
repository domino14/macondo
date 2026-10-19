---
id: leave
priority: 40
when: leave_matters
---
**Leave.** The tiles left on the rack after a play are often the reason one
play beats another. You generally want a rough balance of vowels and
consonants - within one of each other counts as balanced - and you want to
avoid keeping too many bad letters. AEILNRST, C, Z and X are good letters; U,
V, Q, J and W are usually bad ones.

Balance matters less when the unseen tiles are lopsided. A leave of three
vowels and no consonants is fine if what's left to draw is consonant-heavy, and
that is worth saying explicitly when it comes up.

Call `get_our_play_metadata` for the exact vowel and consonant counts of a
leave - do not count or estimate them yourself. Call `evaluate_leave` for what
the leave is worth: a leave isn't good until it's worth about +2 to +3, a
really strong one is +8 or above, and a negative value is a poor leave.
