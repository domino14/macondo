package shell

import (
	"embed"
	"errors"
)

var (
	//go:embed helptext
	helptext embed.FS
)

func usage(mode string) (*Response, error) {
	dat, err := helptext.ReadFile("helptext/usage-" + mode + ".txt")
	if err != nil {
		return nil, errors.New("Could not load helptext: " + err.Error())
	}
	return msg(string(dat)), nil
}

func usageTopic(topic string) (*Response, error) {
	path := "helptext/" + topic + ".txt"
	dat, err := helptext.ReadFile(path)
	if err != nil {
		return nil, errors.New("There is no help text for the topic " + topic)
	}
	return msg(string(dat)), nil
}
