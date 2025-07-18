package bot

import "github.com/domino14/macondo/gen/api/proto/macondo"

func hasSimming(botCode macondo.BotRequest_BotCode, botSpec *macondo.BotSpec) bool {
	switch botCode {
	case macondo.BotRequest_SIMMING_BOT,
		macondo.BotRequest_SIMMING_INFER_BOT:
		return true

	case macondo.BotRequest_CUSTOM_BOT:
		if botSpec != nil && botSpec.Params.HasSimming {
			return true
		}
	}
	return false
}

func HasPreendgame(botCode macondo.BotRequest_BotCode, botSpec *macondo.BotSpec) bool {
	switch botCode {
	case macondo.BotRequest_SIMMING_BOT,
		macondo.BotRequest_HASTY_PLUS_ENDGAME_BOT,
		macondo.BotRequest_SIMMING_INFER_BOT:

		return true
	case macondo.BotRequest_CUSTOM_BOT:
		if botSpec != nil && botSpec.Params.HasPreendgame {
			return true
		}
	}
	return false
}

func HasEndgame(botCode macondo.BotRequest_BotCode, botSpec *macondo.BotSpec) bool {
	switch botCode {
	case macondo.BotRequest_SIMMING_BOT,
		macondo.BotRequest_HASTY_PLUS_ENDGAME_BOT,
		macondo.BotRequest_SIMMING_INFER_BOT:

		return true
	case macondo.BotRequest_CUSTOM_BOT:
		if botSpec != nil && botSpec.Params.HasEndgame {
			return true
		}
	}
	return false
}

func HasInfer(botCode macondo.BotRequest_BotCode, botSpec *macondo.BotSpec) bool {
	switch botCode {
	case macondo.BotRequest_SIMMING_INFER_BOT:
		return true

	case macondo.BotRequest_CUSTOM_BOT:
		if botSpec != nil && botSpec.Params.HasInfer {
			return true
		}
	}
	return false
}
