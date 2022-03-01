package shell

import (
	"errors"
	"strings"

	"github.com/rs/zerolog/log"
	lua "github.com/yuin/gopher-lua"
)

func getShell(L *lua.LState) *ShellController {
	shell := L.GetGlobal("macondo_shell")
	ud, ok := shell.(*lua.LUserData)
	if !ok {
		panic("luserdata not right type")
	}
	sc, ok := ud.Value.(*ShellController)
	if !ok {
		panic("shellcontroller not right type")
	}
	return sc
}

func Load(L *lua.LState) int {
	lv := L.ToString(1)
	sc := getShell(L)
	r, err := sc.load(&shellcmd{
		cmd:  "load",
		args: strings.Split(lv, " "),
	})
	if err != nil {
		log.Err(err).Msg("error-executing-load")
		return 0
	}
	L.Push(lua.LString(r.message))
	// return number of results pushed to stack.
	return 1
}

func Gen(L *lua.LState) int {
	lv := L.ToString(1)
	sc := getShell(L)
	r, err := sc.generate(&shellcmd{
		cmd:  "gen",
		args: strings.Split(lv, " "),
	})
	if err != nil {
		log.Err(err).Msg("error-executing-gen")
		return 0
	}
	L.Push(lua.LString(r.message))
	return 1
}

func (sc *ShellController) script(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("need arguments for script")
	}

	filepath := cmd.args[0]

	L := lua.NewState()
	defer L.Close()

	lsc := L.NewUserData()
	lsc.Value = sc

	L.SetGlobal("macondo_shell", lsc)
	L.SetGlobal("macondo_gen", L.NewFunction(Gen))
	L.SetGlobal("macondo_load", L.NewFunction(Load))

	if err := L.DoFile(filepath); err != nil {
		log.Err(err).Msg("there was a error")
		return nil, err
	}
	return nil, nil
}
