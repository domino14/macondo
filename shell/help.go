package shell

import (
	"errors"
	"io/ioutil"
	"path/filepath"
)

func usage(mode string, execPath string) (*Response, error) {
	path := filepath.Join(execPath, "./shell/helptext/usage-"+mode+".txt")
	dat, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, errors.New("Could not load helptext: " + err.Error())
	}
	return msg(string(dat)), nil
}

func usageTopic(topic string, execPath string) (*Response, error) {
	path := filepath.Join(execPath, "./shell/helptext/"+topic+".txt")
	dat, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, errors.New("There is no help text for the topic " + topic)
	}
	return msg(string(dat)), nil
}
