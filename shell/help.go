package shell

import (
	"io"
	"io/ioutil"
	"path/filepath"
)

func usage(w io.Writer, mode string, execPath string) {

	dat, err := ioutil.ReadFile(
		filepath.Join(execPath, "./shell/helptext/usage-"+mode+".txt"))
	if err != nil {
		io.WriteString(w, "Error loading helptext: "+err.Error())
		return
	}
	io.WriteString(w, string(dat))

}

func usageTopic(w io.Writer, topic string, execPath string) {
	dat, err := ioutil.ReadFile(
		filepath.Join(execPath, "./shell/helptext/"+topic+".txt"))

	if err != nil {
		io.WriteString(w, "There is no help text for the topic "+topic+"\n")
		return
	}
	io.WriteString(w, string(dat))
}
