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

func Set(L *lua.LState) int {
	lv := L.ToString(1)
	sc := getShell(L)
	r, err := sc.set(&shellcmd{
		cmd:  "set",
		args: strings.Split(lv, " "),
	})
	if err != nil {
		log.Err(err).Msg("error-executing-set")
		L.Push(lua.LString("ERROR: " + err.Error()))
		return 1
	}
	L.Push(lua.LString(r.message))
	// return number of results pushed to stack.
	return 1
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
		L.Push(lua.LString("ERROR: " + err.Error()))
		return 1
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

func Turn(L *lua.LState) int {
	lv := L.ToString(1)
	sc := getShell(L)
	r, err := sc.turn(&shellcmd{
		cmd:  "turn",
		args: strings.Split(lv, " "),
	})
	if err != nil {
		log.Err(err).Msg("error-executing-turn")
		return 0
	}
	L.Push(lua.LString(r.message))
	return 1
}

func Gid(L *lua.LState) int {
	sc := getShell(L)
	r, err := sc.gid(&shellcmd{
		cmd: "gid",
	})
	if err != nil {
		log.Err(err).Msg("error-executing-gid")
		return 0
	}
	L.Push(lua.LString(r.message))
	return 1
}

func Endgame(L *lua.LState) int {
	lv := L.ToString(1)
	sc := getShell(L)
	cmd, err := extractFields("endgame " + lv)
	if err != nil {
		log.Err(err).Msg("error-parsing-endgame")
		return 0
	}
	r, err := sc.endgame(cmd)
	if err != nil {
		log.Err(err).Msg("error-executing-endgame")
		return 0
	}
	L.Push(lua.LString(r.message))
	return 1
}

func Sim(L *lua.LState) int {
	lv := L.ToString(1)
	sc := getShell(L)
	cmd, err := extractFields("sim " + lv)
	if err != nil {
		log.Err(err).Msg("error-parsing-sim")
		return 0
	}
	_, err = sc.sim(cmd)
	if err != nil {
		log.Err(err).Msg("error-executing-sim")
		return 0
	}

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
	L.SetGlobal("macondo_gid", L.NewFunction(Gid))
	L.SetGlobal("macondo_set", L.NewFunction(Set))
	L.SetGlobal("macondo_turn", L.NewFunction(Turn))
	L.SetGlobal("macondo_endgame", L.NewFunction(Endgame))
	L.SetGlobal("macondo_sim", L.NewFunction(Sim))

	if err := L.DoFile(filepath); err != nil {
		log.Err(err).Msg("there was a error")
		return nil, err
	}
	return nil, nil
}
