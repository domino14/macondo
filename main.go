package main

import (
	"context"
	"flag"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"
	"os/signal"
	"runtime/pprof"
	"time"

	"github.com/domino14/macondo/anagrammer"
	"github.com/domino14/macondo/gaddagmaker"
	"github.com/gorilla/rpc/v2"
	"github.com/gorilla/rpc/v2/json2"
)

const (
	// BlankQuestionsTimeout - how much time to give blank challenge
	// generator before giving up
	BlankQuestionsTimeout = 5000 * time.Millisecond
	// BuildQuestionsTimeout - how much time to give build challenge
	// generator before giving up
	BuildQuestionsTimeout = 5000 * time.Millisecond
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

func addTimeout(i *rpc.RequestInfo) *http.Request {
	var timeout time.Duration
	var ctx context.Context
	shouldModify := false
	switch i.Method {
	case "AnagramService.BlankChallenge":
		timeout = BlankQuestionsTimeout
		shouldModify = true
	case "AnagramService.BuildChallenge":
		timeout = BuildQuestionsTimeout
		shouldModify = true
	}
	if shouldModify {
		// It's ok to not call cancel here (actually i'm not able to)
		// when timeout expires cancel is implicitly called.
		ctx, _ = context.WithTimeout(context.Background(), timeout)
		return i.Request.WithContext(ctx)
	}
	return i.Request
}

var dawgPath = flag.String("dawgpath", "", "path for dawgs")
var profilePath = flag.String("profilepath", "", "path for profile")
var noproxy = flag.Bool("noproxy", false,
	"set this to true if running locally (no reverse proxy)")

func main() {
	flag.Parse()
	anagrammer.LoadDawgs(*dawgPath)

	if *profilePath != "" {
		f, err := os.Create(*profilePath)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	appendPath := ""
	if *noproxy {
		appendPath = "/macondo"
	}

	http.HandleFunc("/", mainHandler)
	http.HandleFunc(appendPath+"/static/", func(w http.ResponseWriter, r *http.Request) {
		pathToServe := r.URL.Path[1+len(appendPath):]
		log.Printf("[DEBUG] Serving file, path=%v", pathToServe)
		http.ServeFile(w, r, pathToServe)
	})
	fmt.Println("Listening on http://localhost:8088/")
	s := rpc.NewServer()
	s.RegisterCodec(json2.NewCodec(), "application/json")
	s.RegisterService(new(gaddagmaker.GaddagService), "")
	s.RegisterService(new(anagrammer.AnagramService), "")
	// Need to set rpc v2 to master to use the following, in the dep toml file :/
	// This allows us to modify the request and optionally add a context
	// timeout.
	s.RegisterInterceptFunc(addTimeout)
	http.Handle(appendPath+"/rpc", s)

	server := &http.Server{Addr: ":8088", Handler: nil}

	go func() {
		if err := server.ListenAndServe(); err != nil {
			// handle err
		}
	}()

	// Setting up signal capturing
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt)

	// Waiting for SIGINT (pkill -2)
	<-stop

	ctx, _ := context.WithTimeout(context.Background(), 5*time.Second)
	if err := server.Shutdown(ctx); err != nil {
		// handle err
	}

	log.Println("Exiting...")
}
