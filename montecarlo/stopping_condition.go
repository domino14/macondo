package montecarlo

type StoppingCondition int

const (
	StopNone StoppingCondition = iota
	Stop90
	Stop95
	Stop98
	Stop99
	Stop999
)

const (
	defaultIterationsCutoff   = 2000
	defaultPerPlyStopScaling  = 625
	defaultMinReasonableWProb = 0.005 // or 0.5%
)

// AutoStopper holds the sim's stopping settings; the stopping rule itself is
// best-arm identification (see bai_sim.go).
type AutoStopper struct {
	// The sim stops after iterationsCutoff + plies*perPlyStopScaling
	// iterations at the most, when a stopping condition is set.
	iterationsCutoff  int
	perPlyStopScaling int

	stoppingCondition StoppingCondition
}

func newAutostopper() *AutoStopper {
	return &AutoStopper{
		iterationsCutoff:  defaultIterationsCutoff,
		perPlyStopScaling: defaultPerPlyStopScaling,
	}
}
