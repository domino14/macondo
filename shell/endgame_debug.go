package shell

import (
	"errors"
	"os"
)

func (sc *ShellController) endgameDebugModeSwitch(line string, sig chan os.Signal) error {

	cmd, err := extractFields(line)
	if err != nil {
		return err
	}

	switch cmd.cmd {
	case "mode":
		out, err := sc.setMode(cmd)
		if err != nil {
			return err
		} else {
			sc.showMessage(out.message)
			return nil
		}

	case "help":
		out, err := usage("endgamedebug")
		if err != nil {
			return err
		} else {
			sc.showMessage(out.message)
			return nil
		}

	// case "l":
	// 	for _, n := range sc.endgameSolver.LastPrincipalVariationNodes() {
	// 		sc.showMessage(n.String())
	// 		sc.showMessage("-----")
	// 	}

	default:
		return errors.New("command not recognized: " + cmd.cmd)
	}
	return nil
}
