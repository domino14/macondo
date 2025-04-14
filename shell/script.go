package shell

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/cjoudrey/gluahttp"
	"github.com/domino14/macondo/gen/api/proto/macondo"
	"github.com/rs/zerolog/log"
	lua "github.com/yuin/gopher-lua"
	luajson "layeh.com/gopher-json"
)

var exports = map[string]lua.LGFunction{
	"load":       load,
	"set":        set,
	"gen":        gen,
	"add":        add,
	"sim":        sim,
	"turn":       turn,
	"gid":        gid,
	"endgame":    endgame,
	"busy":       busy,
	"elite_play": elitePlay,
	"last":       last,
	"gamestate":  gamestate,
}

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

func set(L *lua.LState) int {
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

func load(L *lua.LState) int {
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

func gamestate(L *lua.LState) int {
	sc := getShell(L)
	r, err := sc.gameState(nil)
	if err != nil {
		log.Err(err).Msg("error-executing-gamestate")
		L.Push(lua.LString("ERROR: " + err.Error()))
		return 1
	}
	L.Push(lua.LString(r.message))
	// return number of results pushed to stack.
	return 1
}

func last(L *lua.LState) int {
	sc := getShell(L)
	r, err := sc.last(nil)
	if err != nil {
		log.Err(err).Msg("error-executing-last")
		L.Push(lua.LString("ERROR: " + err.Error()))
		return 1
	}
	L.Push(lua.LString(r.message))
	// return number of results pushed to stack.
	return 1
}

func gen(L *lua.LState) int {
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

func add(L *lua.LState) int {
	lv := L.ToString(1)
	sc := getShell(L)
	cmd, err := extractFields("add " + lv)
	if err != nil {
		log.Err(err).Msg("error-parsing-add")
		return 0
	}
	_, err = sc.add(cmd)
	if err != nil {
		log.Err(err).Msg("error-executing-add")
		return 0
	}
	return 1
}

func turn(L *lua.LState) int {
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

func gid(L *lua.LState) int {
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

func elitePlay(L *lua.LState) int {
	sc := getShell(L)
	sc.botCtx, sc.botCtxCancel = context.WithTimeout(context.Background(), time.Second*time.Duration(60))
	defer sc.botCtxCancel()

	if sc.elitebot.History().PlayState == macondo.PlayState_GAME_OVER {
		log.Error().Msg("game is over")
		return 0
	}

	m, err := sc.elitebot.BestPlay(sc.botCtx)
	if err != nil {
		log.Err(err).Msg("error with eliteplay")
		return 0
	}
	L.Push(lua.LString(sc.game.Board().MoveDescriptionWithPlaythrough(m)))
	L.Push(lua.LString(sc.elitebot.BestPlayDetails(sc.botCtx)))
	return 2
}

func endgame(L *lua.LState) int {
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

func sim(L *lua.LState) int {
	lv := L.ToString(1)
	sc := getShell(L)
	cmd, err := extractFields("sim " + lv)
	if err != nil {
		log.Err(err).Msg("error-parsing-sim")
		return 0
	}
	resp, err := sc.sim(cmd)
	if err != nil {
		log.Err(err).Msg("error-executing-sim")
		return 0
	}
	if resp != nil {
		L.Push(lua.LString(resp.message))
		return 1
	}
	return 0
}

func busy(L *lua.LState) int {
	sc := getShell(L)
	L.Push(lua.LBool(sc.solving()))
	return 1
}

func Loader(L *lua.LState) int {
	mod := L.SetFuncs(L.NewTable(), exports)

	L.Push(mod)
	return 1
}

func (sc *ShellController) script(cmd *shellcmd) (*Response, error) {
	if cmd.args == nil {
		return nil, errors.New("need arguments for script")
	}
	if sc.solving() {
		return nil, errMacondoSolving
	}

	filepath := cmd.args[0]

	L := lua.NewState()
	defer L.Close()

	L.PreloadModule("macondo", Loader)
	L.PreloadModule("http", gluahttp.NewHttpModule(&http.Client{}).Loader)
	luajson.Preload(L)

	lsc := L.NewUserData()
	lsc.Value = sc

	L.SetGlobal("macondo_shell", lsc)

	if len(cmd.args) > 1 {
		table := L.NewTable()
		joinedStr := strings.Join(cmd.args[1:], " ")
		table.Insert(1, lua.LString(joinedStr))
		L.SetGlobal("args", table)
	}

	if err := L.DoFile(filepath); err != nil {
		log.Err(err).Msg("there was a error")
		return nil, err
	}
	return nil, nil
}
