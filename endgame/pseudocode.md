```
func solve_endgame(myrack, opprack):
    mymoves = gen_all(myrack, sortby=score)
    oppmoves = gen_all(opprack, sortby=score)

    tree = {}
    for move in mymoves:
        move.evaluate()
        # Attach all oppmoves as descendants of `move` in tree
        # Note we don't care about the move being blocked right now.
        tree_attach(move, oppmoves)


func (m *move) evaluate():
    # Attach a pessimistic and optimistic evaluation of this move.



```