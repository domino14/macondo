package game

import (
	"fmt"
	"strings"
	"sync"

	"github.com/rs/zerolog/log"

	"github.com/domino14/word-golib/tilemapping"

	"github.com/domino14/macondo/config"
	"github.com/domino14/macondo/dataloaders"
	"github.com/domino14/macondo/triton"
)

// Each lexicon family can have its own net. triton-model-name (or a bot's
// own model) is a base name; a game plays with <base>-<suffix> when that
// model exists, where the suffix is "csw" for the Collins lexica, the
// letter distribution's name for other alphabets ("german", "french", ...),
// and none for the other English lexica. An English game falls back to the
// base model (the NWL net plays CSW well); a game in another alphabet has no
// fallback, since a net only knows the alphabet it was trained on.

// MLModelSuffix is the model-name suffix for games in this lexicon, "" for
// English lexica other than Collins.
func MLModelSuffix(lexicon string, ld *tilemapping.LetterDistribution) string {
	if !strings.EqualFold(ld.Name, "english") {
		return strings.ToLower(ld.Name)
	}
	if strings.HasPrefix(strings.ToUpper(lexicon), "CSW") {
		return "csw"
	}
	return ""
}

type mlModelKey struct {
	base, lexicon, dist, version, server string
}

var mlModelNames sync.Map // mlModelKey -> string

// ResolveMLModel is the model a game in this lexicon plays with, given the
// base name: <base>-<suffix> if it exists, else the base for English, else
// an error.
func ResolveMLModel(cfg *config.Config, base, lexicon string, ld *tilemapping.LetterDistribution) (string, error) {
	suffix := MLModelSuffix(lexicon, ld)
	if suffix == "" || strings.HasSuffix(base, "-"+suffix) {
		return base, nil
	}
	useTriton := cfg.GetBool(config.ConfigTritonUseTriton)
	ver := mlModelVersion(cfg)
	key := mlModelKey{base, lexicon, ld.Name, ver, "onnx"}
	if useTriton {
		key.server = cfg.GetString(config.ConfigTritonURL)
	}
	if name, ok := mlModelNames.Load(key); ok {
		return name.(string), nil
	}
	want := base + "-" + suffix
	var found bool
	if useTriton {
		var err error
		found, err = triton.InRepository(key.server, want)
		if err != nil {
			log.Warn().Err(err).Str("model", want).Msg("could not list Triton's models")
		}
	} else {
		file := fmt.Sprintf("models/%s/%s/model.onnx", want, ver)
		if r, err := dataloaders.StratFileForLexicon(dataloaders.StrategyParamsPath(cfg.WGLConfig()), file, lexicon); err == nil {
			r.Close()
			found = true
		}
	}
	name := want
	if !found {
		if !strings.EqualFold(ld.Name, "english") {
			return "", fmt.Errorf("no %s model for %s: looked for %s", ld.Name, lexicon, want)
		}
		name = base
	}
	log.Info().Str("lexicon", lexicon).Str("model", name).Msg("ml-model-for-lexicon")
	mlModelNames.Store(key, name)
	return name, nil
}

func mlModelVersion(cfg *config.Config) string {
	if v := cfg.GetString(config.ConfigTritonModelVersion); v != "" {
		return v
	}
	return "1"
}

// ResolveMLModel is the model this game plays with, given a base name.
func (g *Game) ResolveMLModel(base string) (string, error) {
	return ResolveMLModel(g.config, base, g.LexiconName(), g.letterDistribution)
}

// MLModelName is the configured model (triton-model-name) for this game's
// lexicon.
func (g *Game) MLModelName() (string, error) {
	return g.ResolveMLModel(g.config.GetString(config.ConfigTritonModelName))
}

// checkMLModelShape errors when a model's inputs (0 if unknown) do not fit
// the layout: a net only reads positions in the alphabet it was trained on.
func checkMLModelShape(model string, shape func() (planes, scalars int), lay *MLLayout, dist string) error {
	planes, scalars := shape()
	if (planes != 0 && planes != lay.NumPlanes) || (scalars != 0 && scalars != lay.NumScalars) {
		return fmt.Errorf("model %s takes %d board planes and %d scalars, but %s positions have %d and %d (%d letters): it was trained on another alphabet",
			model, planes, scalars, dist, lay.NumPlanes, lay.NumScalars, lay.Letters)
	}
	return nil
}
