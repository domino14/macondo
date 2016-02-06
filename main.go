package main

import (
	"flag"
	"fmt"
	"github.com/domino14/macondo/anagrammer"
	"github.com/domino14/macondo/gaddag"
	"github.com/gorilla/rpc/v2"
	"github.com/gorilla/rpc/v2/json2"
	"html/template"
	"log"
	"net/http"
	"os"
	"runtime/pprof"
)

var templates = template.Must(template.ParseFiles(
	"templates/index.html"))

func renderTemplate(w http.ResponseWriter, tmpl string) {
	err := templates.ExecuteTemplate(w, tmpl+".html", nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func mainHandler(w http.ResponseWriter, r *http.Request) {
	renderTemplate(w, "index")
}

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
var dawgPath = flag.String("dawgpath", "", "path for dawgs")

func main() {
	flag.Parse()
	anagrammer.LoadDawgs(*dawgPath)
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		bcArgs := &anagrammer.BlankChallengeArgs{
			WordLength: 8, NumQuestions: 25, Lexicon: "America", MaxSolutions: 5,
			Num2Blanks: 2,
		}
		anagrammer.GenerateBlanks(bcArgs, anagrammer.Dawgs["America"])
		defer pprof.StopCPUProfile()
		return
	}
	http.HandleFunc("/", mainHandler)
	http.HandleFunc("/static/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, r.URL.Path[1:])
	})
	fmt.Println("Listening on http://localhost:8088/")
	s := rpc.NewServer()
	s.RegisterCodec(json2.NewCodec(), "application/json")
	s.RegisterService(new(gaddag.GaddagService), "")
	s.RegisterService(new(anagrammer.AnagramService), "")
	http.Handle("/rpc", s)
	err := http.ListenAndServe(":8088", nil)
	if err != nil {
		log.Fatalln(err)
	}

}
