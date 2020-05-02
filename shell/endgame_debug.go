package shell

import (
	"errors"
	"fmt"
	"os"
	"strconv"
)

func (sc *ShellController) endgameDebugModeSwitch(line string, sig chan os.Signal) error {
	cmd, args := extractFields(line)

	switch cmd {
	case "mode":
		if args == nil {
			sc.showError(errors.New("select a mode"))
			break
		}
		sc.modeSelector(args[0])

	case "help":
		usage(sc.l.Stderr(), "endgamedebug")

	case "l":
		// List the current level of nodes
		if sc.curEndgameNode == nil {
			sc.curEndgameNode = sc.endgameSolver.RootNode()
		}

		sc.showMessage(sc.curEndgameNode.String())
		sc.showMessage("Children:")
		sc.showMessage("----------------")

		for i, c := range sc.curEndgameNode.Children() {
			sc.showMessage(fmt.Sprintf("%v: %v", i, c.String()))
		}
		sc.showMessage("----------------")

	case "u":
		// List the current level of nodes
		if sc.curEndgameNode == nil {
			sc.showMessage("Can't go up any farther")
			break
		}
		sc.curEndgameNode = sc.curEndgameNode.Parent()
		if sc.curEndgameNode != nil {
			sc.showMessage(sc.curEndgameNode.String())
		} else {
			sc.showMessage("<nil>")
		}

	case "i":
		// List info about the current node.
		if sc.curEndgameNode != nil {
			sc.showMessage(sc.curEndgameNode.String())
		} else {
			sc.showMessage("<nil>")
		}

	case "s":
		if len(args) == 0 {
			sc.showError(errors.New("select a node to step into"))
			break
		}
		nodeID, err := strconv.Atoi(args[0])
		if err != nil {
			sc.showError(err)
			return nil
		}
		if nodeID >= len(sc.curEndgameNode.Children()) || nodeID < 0 {
			sc.showError(errors.New("index not in range"))
			return nil
		}

		sc.showMessage(fmt.Sprintf("Stepping into child %d", nodeID))
		sc.curEndgameNode = sc.curEndgameNode.Children()[nodeID]

	default:
		sc.showError(errors.New("command not recognized: " + cmd))
	}
	return nil
}
