package kwg

import (
	"strings"
	"testing"

	"github.com/domino14/macondo/tilemapping"
	"github.com/matryer/is"
)

func BenchmarkAnagramBlanks(b *testing.B) {
	// ~0.62 ms on 12thgen-monolith
	is := is.New(b)
	kwg, err := Get(&DefaultConfig, "CSW21")
	is.NoErr(err)
	alph := kwg.GetAlphabet()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var anags []string
		da := KWGAnagrammer{}
		if err = da.InitForString(kwg, "RETINA??"); err != nil {
			b.Error(err)
		} else if err = da.Anagram(kwg, func(word tilemapping.MachineWord) error {
			anags = append(anags, word.UserVisible(alph))
			return nil
		}); err != nil {
			b.Error(err)
		}
	}
}

func TestAnagramBlanks(t *testing.T) {
	is := is.New(t)
	d, err := Get(&DefaultConfig, "CSW21")
	is.NoErr(err)
	alph := d.GetAlphabet()

	var anags []string
	da := KWGAnagrammer{}
	if err = da.InitForString(d, "RETINA??"); err != nil {
		t.Error(err)
	} else if err = da.Anagram(d, func(word tilemapping.MachineWord) error {
		anags = append(anags, word.UserVisible(alph))
		return nil
	}); err != nil {
		t.Error(err)
	}
	is.Equal(anags, strings.Split(
		"ACENTRIC ACTIONER AERATING AERATION ALERTING ALTERING ANGRIEST ANGSTIER ANIMATER ANKERITE ANOESTRI ANOINTER ANORETIC ANTERIOR ANTHERID ANTIHERO ANTIMERE ANTIQUER ANTIRAPE ANTISERA ANTIWEAR ANURETIC APERIENT ARENITES ARENITIC ARETTING ARGENTIC AROINTED ARSENITE ARSONITE ARTESIAN ARTINESS ASTRINGE ATABRINE ATEBRINS ATHERINE ATRAZINE ATROPINE ATTAINER AUNTLIER AVERTING BACTERIN BANISTER BARITONE BARNIEST BERATING BRAUNITE CANISTER CARINATE CARNIEST CATERING CENTIARE CERATINS CISTERNA CITRANGE CLARINET CRANIATE CREATINE CREATING CREATINS CREATION CRINATED DAINTIER DATURINE DENTARIA DERATING DERATION DETAINER DETRAINS DICENTRA DIPTERAN EARTHING ELATERIN EMIGRANT ENARGITE ENTAILER ENTRAILS ENTRAINS EXPIRANT FAINTERS FAINTIER FENITARS GANISTER GANTRIES GNATTIER GRADIENT GRANITES GRATINEE GRIEVANT HAIRNETS HAURIENT HEARTING HERNIATE INAURATE INCREATE INDARTED INDURATE INEARTHS INERRANT INERTIAE INERTIAL INERTIAS INFLATER INGATHER INGRATES INORNATE INTEGRAL INTERACT INTERAGE INTERLAP INTERLAY INTERMAT INTERNAL INTERVAL INTERWAR INTRANET INTREATS ITERANCE JAUNTIER KERATINS KNITWEAR KREATINE LARNIEST LATRINES MARINATE MARTINET MERANTIS MINARETS NAARTJIE NACRITES NARKIEST NARTJIES NAVICERT NITRATED NITRATES NOTAIRES NOTARIES NOTARISE NOTARIZE OBTAINER ORDINATE ORIENTAL PAINTERS PAINTIER PAINTURE PANTRIES PERIANTH PERTAINS PINASTER PRETRAIN PRISTANE QUAINTER RABATINE RAIMENTS RAINDATE RAINIEST RANDIEST RANGIEST RATANIES RATIONED RATLINES RATTLINE REACTING REACTION REANOINT REASTING REATTAIN REBATING REDATING REINSTAL RELATING RELATION REMATING REOBTAIN REPAINTS RESIANTS RESINATA RESINATE RESTRAIN RETAINED RETAINER RETAKING RETAPING RETAXING RETINALS RETINULA RETIRANT RETRAINS RETSINAS ROSINATE RUINATED RUINATES RUMINATE SANTERIA SATINIER SCANTIER SEATRAIN SENORITA SLANTIER SNARIEST STAINERS STARNIES STEARINE STEARING STEARINS STRAINED STRAINER STRAITEN TABERING TABORINE TACRINES TAILERON TAINTURE TANGLIER TAPERING TARTINES TARWHINE TASERING TAURINES TAVERING TENTORIA TENURIAL TERAGLIN TERMINAL TERRAINS TERRAPIN TERTIANS THERIANS TINWARES TRAINEES TRAINERS TRAINMEN TRAMLINE TRANCIER TRANNIES TRANSIRE TRAPLINE TREADING TREATING TREENAIL TRENAILS TRIANGLE TRIAZINE TRIENNIA TRIPLANE TRIPTANE TWANGIER TYRAMINE URANITES URBANITE URINATED URINATES VAUNTIER VAWNTIER VERATRIN VINTAGER WARIMENT WATERING XERANTIC", " "))
	is.Equal(len(anags), 259)
}

func TestAnagram(t *testing.T) {
	is := is.New(t)
	d, err := Get(&DefaultConfig, "CSW21")
	is.NoErr(err)
	alph := d.GetAlphabet()

	var anags []string
	da := KWGAnagrammer{}
	if err = da.InitForString(d, "AZ"); err != nil {
		t.Error(err)
	} else if err = da.Anagram(d, func(word tilemapping.MachineWord) error {
		anags = append(anags, word.UserVisible(alph))
		return nil
	}); err != nil {
		t.Error(err)
	}
	is.Equal(anags, []string{"ZA"})
}

type testpair struct {
	prefix string
	found  bool
}

var findWordTests = []testpair{
	{"ZYZZ", false},
	{"ZYZZZ", false},
	{"BAREFIT", true},
	{"KWASH", false},
	{"KWASHO", false},
	{"BAREFITS", false},
	{"AASVOGELATION", false},
	{"FIREFANGNESS", false},
	{"TRIREMED", false},
	{"BAREFITD", false},
	{"KAFF", false},
	{"FF", false},
	{"ABRACADABRA", true},
	{"EEE", false},
	{"ABC", false},
	{"ABCD", false},
	{"FIREFANG", true},
	{"X", false},
	{"Z", false},
	{"Q", false},
	{"KWASHIORKORS", true},
	{"EE", false},
	{"RETIARII", true},
	{"CINEMATOGRAPHER", true},
	{"ANIMADVERTS", true},
	{"PRIVATDOZENT", true},
	{"INEMATOGRAPHER", false},
	{"RIIRAITE", false},
	{"GG", false},
	{"LL", false},
	{"ZZ", false},
	{"ZZZ", true},
	{"ZZZS", false},
}

func TestFindMachineWord(t *testing.T) {
	is := is.New(t)
	d, err := Get(&DefaultConfig, "NWL20")
	is.NoErr(err)
	l := Lexicon{KWG: *d}

	for _, pair := range findWordTests {
		t.Run(pair.prefix, func(t *testing.T) {
			mw, err := tilemapping.ToMachineLetters(pair.prefix, d.GetAlphabet())
			is.NoErr(err)
			found := l.HasWord(mw)
			is.Equal(found, pair.found)
		})
	}
}

var findNorwegianWordTests = []testpair{
	{"ABACAER", true},
	{"ÅMA", true},
	{"AMÅ", false},
	{"ÜBERKUL", true}, // takes an E and a T!
}

func TestFindMachineWordNorwegian(t *testing.T) {
	is := is.New(t)
	d, err := Get(&DefaultConfig, "NSF22")
	is.NoErr(err)
	l := Lexicon{KWG: *d}

	for _, pair := range findNorwegianWordTests {
		t.Run(pair.prefix, func(t *testing.T) {
			mw, err := tilemapping.ToMachineLetters(pair.prefix, d.GetAlphabet())
			is.NoErr(err)
			found := l.HasWord(mw)
			is.Equal(found, pair.found)
		})
	}
}

var hasAnagramNorwegianTests = []testpair{
	{"BRACEAA", true},
	{"MÅA", true},
	{"AMÅA", false},
	{"BELRÜUK", true},
	{"BELRSÜUK", false},
	{"BELRTÜUK", true},
}

func TestHasAnagramNorwegian(t *testing.T) {
	is := is.New(t)
	d, err := Get(&DefaultConfig, "NSF22")
	is.NoErr(err)
	l := Lexicon{KWG: *d}

	for _, pair := range hasAnagramNorwegianTests {
		t.Run(pair.prefix, func(t *testing.T) {
			mw, err := tilemapping.ToMachineLetters(pair.prefix, d.GetAlphabet())
			is.NoErr(err)
			found := l.HasAnagram(mw)
			is.Equal(found, pair.found)
		})
	}
}

func TestHasAnagramEnglish(t *testing.T) {
	is := is.New(t)
	d, err := Get(&DefaultConfig, "CSW21")
	is.NoErr(err)
	l := Lexicon{KWG: *d}

	mw, err := tilemapping.ToMachineLetters("AZ", d.GetAlphabet())
	is.NoErr(err)
	found := l.HasAnagram(mw)
	is.Equal(found, true)
}
