package main

import (
	"fmt"
	"github.com/domino14/macondo/anagrammer"
	"github.com/domino14/macondo/gaddag"
	"github.com/gorilla/rpc/v2"
	"github.com/gorilla/rpc/v2/json2"
	"html/template"
	"net/http"
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

func main() {
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
	http.ListenAndServe(":8088", nil)

}
