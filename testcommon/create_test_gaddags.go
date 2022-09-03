package testcommon

import (
	"os"
	"path/filepath"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/gaddagmaker"
)

func CreateGaddags(cfg config.Config, lexica []string) {
	for _, lex := range lexica {
		gdgPath := filepath.Join(cfg.LexiconPath, "gaddag", lex+".gaddag")
		if _, err := os.Stat(gdgPath); os.IsNotExist(err) {
			gaddagmaker.GenerateGaddag(filepath.Join(cfg.LexiconPath, lex+".txt"), true, true)
			err = os.Rename("out.gaddag", gdgPath)
			if err != nil {
				panic(err)
			}
		}
	}
}

func CreateDawgs(cfg config.Config, lexica []string) {
	for _, lex := range lexica {
		gdgPath := filepath.Join(cfg.LexiconPath, "dawg", lex+".dawg")
		if _, err := os.Stat(gdgPath); os.IsNotExist(err) {
			gaddagmaker.GenerateDawg(filepath.Join(cfg.LexiconPath, lex+".txt"), true, true, false)
			err = os.Rename("out.dawg", gdgPath)
			if err != nil {
				panic(err)
			}
		}
	}
}
