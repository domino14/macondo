package shell

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/domino14/macondo/config"
)

var quitting bool

func UCGILoop(cfg *config.Config) {
	// we're using the shell for its helper structures/functions only.
	sc := NewShellController(cfg)
	sc.SetMode(ModeUCGI)

	quitting = false
	scanner := bufio.NewScanner(os.Stdin)
	for !quitting {
		if !scanner.Scan() {
			break // Exit loop if input ends
		}
		command := scanner.Text()
		err := processCommand(sc, command)
		if err != nil {
			errout(err)
		}
	}
}

func errout(err error) {
	fmt.Println("error", err.Error())
}

func processCommand(sc *ShellController, command string) error {
	cmd, err := extractFields(command)
	if err != nil {
		return err
	}
	switch cmd.cmd {
	case "ucgi":
		fmt.Println("ucgiok")
	case "quit":
		quitting = true
		return nil
	case "cgp":

		if len(cmd.args) < 1 {
			return errors.New("need to provide a cgp string")
		}
		cgpStr := strings.Join(cmd.args[0:], " ")
		err = sc.loadCGP(cgpStr)
		if err != nil {
			return err
		}

	case "gen":
		resp, err := sc.generate(cmd)
		if err != nil {
			return err
		}
		fmt.Printf(resp.message)
	case "sim":

	}
	return nil
}
