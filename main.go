package main

import (
	"context"
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/domino14/macondo/anagrammer"
	"github.com/domino14/macondo/gaddag"
	"github.com/gorilla/rpc/v2"
	"github.com/gorilla/rpc/v2/json2"
)

const (
	// BlankQuestionsTimeout - how much time to give blank challenge
	// generator before giving up
	BlankQuestionsTimeout = 5000 * time.Millisecond
	// BuildQuestionsTimeout - how much time to give build challenge
	// generator before giving up
	BuildQuestionsTimeout = 10000 * time.Millisecond
)

var templates = template.Must(template.ParseFiles(
	"templates/index.html"))
var AuthorizationKey = os.Getenv("AUTH_KEY")

type middleware func(next http.HandlerFunc) http.HandlerFunc

func renderTemplate(w http.ResponseWriter, tmpl string) {
	err := templates.ExecuteTemplate(w, tmpl+".html", nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func mainHandler(w http.ResponseWriter, r *http.Request) {
	renderTemplate(w, "index")
}

func init() {
	if AuthorizationKey == "" {
		panic("No auth key defined")
	}
}

var dawgPath = flag.String("dawgpath", "", "path for dawgs")

func withOptionalAuth(next http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Authorization-Key") != AuthorizationKey {
			http.Error(w, "missing key", http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	}
}

func addTimeout(i *rpc.RequestInfo) *http.Request {
	var timeout time.Duration
	shouldModify := false
	switch i.Method {
	case "AnagramService.BlankChallenge":
		timeout = 1500 * time.Millisecond
		shouldModify = true
	case "AnagramService.BuildChallenge":
		log.Printf("Setting timeout to 5000 ms")
		timeout = 5000 * time.Millisecond
		shouldModify = true

	}
	if shouldModify {
		ctx, _ := context.WithTimeout(context.Background(), timeout)
		return i.Request.WithContext(ctx)
	}
	return i.Request
}

func main() {
	flag.Parse()
	anagrammer.LoadDawgs(*dawgPath)

	http.HandleFunc("/", mainHandler)
	http.HandleFunc("/static/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, r.URL.Path[1:])
	})
	fmt.Println("Listening on http://localhost:8088/")
	s := rpc.NewServer()
	s.RegisterCodec(json2.NewCodec(), "application/json")
	s.RegisterService(new(gaddag.GaddagService), "")
	s.RegisterService(new(anagrammer.AnagramService), "")
	// Need to set rpc v2 to master to use the following, in the dep toml file :/
	// This allows us to modify the request and optionally add a context
	// timeout.
	s.RegisterInterceptFunc(addTimeout)
	http.Handle("/rpc", withOptionalAuth(s))
	err := http.ListenAndServe(":8088", nil)
	if err != nil {
		log.Fatalln(err)
	}

}
