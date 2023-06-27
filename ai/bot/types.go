package bot

import "github.com/domino14/macondo/gen/api/proto/macondo"

func hasSimming(botCode macondo.BotRequest_BotCode) bool {
	switch botCode {
	case macondo.BotRequest_SIMMING_BOT,
		macondo.BotRequest_SIMMING_INFER_BOT,
		macondo.BotRequest_SIMMING_AB:
		return true
	}
	return false
}

func hasPreendgame(botCode macondo.BotRequest_BotCode) bool {
	switch botCode {
	case macondo.BotRequest_SIMMING_BOT,
		macondo.BotRequest_HASTY_PLUS_ENDGAME_BOT,
		macondo.BotRequest_SIMMING_INFER_BOT,
		macondo.BotRequest_SIMMING_AB:

		return true
	}
	return false
}

func hasEndgame(botCode macondo.BotRequest_BotCode) bool {
	switch botCode {
	case macondo.BotRequest_SIMMING_BOT,
		macondo.BotRequest_HASTY_PLUS_ENDGAME_BOT,
		macondo.BotRequest_SIMMING_INFER_BOT,
		macondo.BotRequest_SIMMING_AB:

		return true
	}
	return false
}

func HasInfer(botCode macondo.BotRequest_BotCode) bool {
	switch botCode {
	case macondo.BotRequest_SIMMING_INFER_BOT:
		return true
	}
	return false
}
