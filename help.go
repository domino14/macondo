package main

import (
	"io"
	"io/ioutil"
)

func usage(w io.Writer) {

	dat, err := ioutil.ReadFile("./helptext/usage.txt")
	if err != nil {
		io.WriteString(w, "Error loading helptext: "+err.Error())
		return
	}
	io.WriteString(w, string(dat))

}

func usageTopic(w io.Writer, topic string) {
	dat, err := ioutil.ReadFile("./helptext/" + topic + ".txt")

	if err != nil {
		io.WriteString(w, "There is no help text for the topic "+topic+"\n")
		return
	}
	io.WriteString(w, string(dat))
}
