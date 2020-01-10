package shell

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/rs/zerolog/log"
)

func (sc *ShellController) endgameDebugModeSwitch(line string, sig chan os.Signal) error {
	switch {
	case strings.HasPrefix(line, "mode"):
		sc.modeSelector(line)

	case strings.HasPrefix(line, "help"):
		if strings.TrimSpace(line) == "help" {
			usage(sc.l.Stderr(), "endgamedebug")
		} else {
			showMessage("No additional info is available for this mode", sc.l.Stderr())
		}

	case line == "l":
		// List the current level of nodes
		sc.curEndgameNode = sc.endgameSolver.RootNode()

		sc.showMessage(sc.curEndgameNode.String())
		sc.showMessage("Children:")
		sc.showMessage("----------------")

		for i, c := range sc.curEndgameNode.Children() {
			sc.showMessage(fmt.Sprintf("%v: %v", i, c.String()))
		}
		sc.showMessage("----------------")

	case line == "u":
		// List the current level of nodes
		sc.curEndgameNode = sc.curEndgameNode.Parent()
		if sc.curEndgameNode != nil {
			sc.showMessage(sc.curEndgameNode.String())
		} else {
			sc.showMessage("<nil>")
		}

	case line == "i":
		// List info about the current node.
		if sc.curEndgameNode != nil {
			sc.showMessage(sc.curEndgameNode.String())
		} else {
			sc.showMessage("<nil>")
		}

	case strings.HasPrefix(line, "s"):
		nodeID, err := strconv.Atoi(line[1:])
		if err != nil {
			sc.showMessage("Error: " + err.Error())
			return nil
		}
		if nodeID >= len(sc.curEndgameNode.Children()) || nodeID < 0 {
			sc.showMessage("Error: index not in range")
			return nil
		}

		sc.showMessage(fmt.Sprintf("Stepping into child %d", nodeID))
		sc.curEndgameNode = sc.curEndgameNode.Children()[nodeID]

	default:
		log.Debug().Msgf("you said: %v", strconv.Quote(line))
	}
	return nil
}
