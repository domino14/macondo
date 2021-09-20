package shell

import (
	"embed"
	"errors"
)

//go:embed helptext/*
var helpContent embed.FS

func usage(mode string, execPath string) (*Response, error) {
	dat, err := helpContent.ReadFile("helptext/usage-" + mode + ".txt")
	if err != nil {
		return nil, errors.New("Could not load helptext: " + err.Error())
	}
	return msg(string(dat)), nil
}

func usageTopic(topic string, execPath string) (*Response, error) {
	dat, err := helpContent.ReadFile("helptext/" + topic + ".txt")
	if err != nil {
		return nil, errors.New("There is no help text for the topic " + topic)
	}
	return msg(string(dat)), nil
}
