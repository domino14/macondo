package strategy

// MachineWordString is what you get when you cast MachineWord to a string,
// i.e. string(mw) for a MachineWord mw. It's not a user-readable string and
// is only meant to be used as a lookup for maps, which don't allow us to use
// MachineWord as a key directly.
// type MachineWordString string

// SynergyAndEV encapsulates synergy, and, well, EV.
type SynergyAndEV struct {
	synergy float64
	ev      float64
}

// SynergyLeaveMap gets created from a csv file of leaves. See the notebooks
// directory for generation code.
type SynergyLeaveMap map[string]SynergyAndEV
