package automatic

// Leave calculation code. Rough methodology:
// 1) Collect rack, score, tiles in bag for many millions of racks (see automatic_utils.go)
// 2) Take a large set of racks, use tiles in bag greater than some number (
//    maybe 5 or so to avoid endgame biases)
// 3) Normalize to average score
// 4) Weigh each rack according to probability
// 5) For each subset of each rack, calculate "expected value" of it
// 6) Take into account leave scores when doing equity calculations
// 7) Repeat steps 1-6 until values converge
