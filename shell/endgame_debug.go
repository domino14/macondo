package shell

import (
	"errors"
	"fmt"
	"os"
	"strconv"
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
		out, err := usage("endgamedebug", sc.execPath)
		if err != nil {
			return err
		} else {
			sc.showMessage(out.message)
			return nil
		}

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
		if len(cmd.args) == 0 {
			return errors.New("select a node to step into")
		}
		nodeID, err := strconv.Atoi(cmd.args[0])
		if err != nil {
			return err
		}
		if nodeID >= len(sc.curEndgameNode.Children()) || nodeID < 0 {
			return errors.New("index not in range")
		}
		sc.showMessage(fmt.Sprintf("Stepping into child %d", nodeID))
		sc.curEndgameNode = sc.curEndgameNode.Children()[nodeID]

	default:
		return errors.New("command not recognized: " + cmd.cmd)
	}
	return nil
}
