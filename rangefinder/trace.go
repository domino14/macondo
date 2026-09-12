package rangefinder

import (
	"github.com/domino14/word-golib/tilemapping"
)

// Tracing what the engine chose to look at, and what it found there.
//
// The measure-impute-recalibrate loop spends its whole budget deciding which
// leaves deserve a mini-sim. Whether it decides well is not visible from the
// posterior it ends up with: a read can be wrong because the model mis-scores
// the leaves it saw, or because it never drew the leave that mattered, and
// those two failures call for opposite fixes. Telling them apart needs the
// draws themselves.
//
// A trace records, for every leave the engine measured, what the model
// believed about it *before* the mini-sim ran. Every refine draw is a genuine
// out-of-sample prediction -- the loop only ever draws leaves it has not
// measured -- so Predicted against Measured is an honest calibration test, and
// Q against Measured says whether the proposal is pointing anywhere useful.
//
// Tracing is off unless a caller asks for it. It costs one small struct per
// measured leaf and nothing at all when disabled.

// DrawRecord is one leave the engine chose to measure.
type DrawRecord struct {
	// Round is 0 for the blind prior-sampled pass, 1..n for refine rounds.
	Round int
	// Leave is the sorted leave, rendered for a human.
	Leave string
	// Q is the proposal probability the round assigned this leave, normalized
	// over that round's candidates. Zero in round 0, which samples the prior
	// directly rather than from a proposal.
	Q float64
	// U is the importance weight P/q the draw carried (1 in round 0).
	U float64
	// Mult is how many of the round's draws landed on this leave.
	Mult int
	// Weight is the posterior weight the leave held going in, normalized so
	// the heaviest leave in the posterior is 1.
	Weight float64
	// Uncertainty is the model's own spread on its estimate for this leave,
	// which is what earns a leave draws it would not get on weight alone.
	Uncertainty float64
	// Predicted is the imputed likelihood the model assigned just before the
	// measurement -- out of sample, since the loop never re-draws a measured
	// leave. Zero in round 0, where there is no model yet.
	Predicted float64
	// Measured is what the mini-sim found: P(the play they made | this leave).
	Measured float64
}

// RoundRecord summarizes one pass of the loop.
type RoundRecord struct {
	Round int
	// Drawn, Distinct and Evaluated are the draws asked for, the distinct
	// leaves they landed on, and the ones that finished in time.
	Drawn, Distinct, Evaluated int
	// LogRatio is log R̂ for the batch: measured mass over predicted mass, the
	// loop's own calibration test. SELogRatio is its standard error.
	LogRatio, SELogRatio float64
	// UnmeasuredMass is the share of posterior mass still unmeasured going in.
	UnmeasuredMass float64
	// Converged is whether the round's ratio test ended the loop.
	Converged bool
	// LogCalib is the cross-fitted calibration constant after the refit, and
	// LogCalibInSample the same constant fitted on the data the model saw.
	// Their gap is how far the imputation overfits: near zero means the model
	// generalizes, large means it only explains what it has already measured.
	LogCalib, LogCalibInSample float64
	// MeasuredMass is the share of posterior mass on measured leaves after
	// the refit.
	MeasuredMass float64
}

// InferenceTrace is the record of one inference's draws and rounds.
type InferenceTrace struct {
	Draws  []DrawRecord
	Rounds []RoundRecord
}

// Trace returns the trace collected for the last inference, or nil when
// tracing was not enabled.
func (r *RangeFinder) Trace() *InferenceTrace { return r.trace }

// SetTracing turns draw recording on or off. It takes effect at the next
// PrepareFinder, which is where the trace is reset.
func (r *RangeFinder) SetTracing(on bool) { r.tracing = on }

// startTrace begins a fresh trace, if tracing is on.
func (r *RangeFinder) startTrace() {
	if r.tracing {
		r.trace = &InferenceTrace{}
	} else {
		r.trace = nil
	}
}

// traceDraw records one measured leave. Safe to call with tracing off.
func (r *RangeFinder) traceDraw(rec DrawRecord, leave []tilemapping.MachineLetter) {
	if r.trace == nil {
		return
	}
	rec.Leave = tilemapping.MachineWord(leave).UserVisible(r.origGame.Alphabet())
	r.trace.Draws = append(r.trace.Draws, rec)
}

// traceRound records one pass of the loop, filling in the calibration figures
// from the refit that has just happened. Safe to call with tracing off.
func (r *RangeFinder) traceRound(rec RoundRecord) {
	if r.trace == nil {
		return
	}
	if res := r.imputeRes; res != nil {
		rec.LogCalib = res.logCalib
		rec.LogCalibInSample = res.logCalibInSample
		rec.MeasuredMass = res.measuredMass
	}
	r.trace.Rounds = append(r.trace.Rounds, rec)
}
