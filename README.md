# macondo

A crossword game move generator, written in Go

Macondo is more than just a simple crossword game move generator! It will have a web interface and later simming capabilities and more.

Current master build status:
[![domino14](https://circleci.com/gh/domino14/macondo.svg?style=svg)](https://circleci.com/gh/domino14/macondo)

# protoc

To generate pb files, run this in the macondo directory:

`protoc --go_out=gen --go_opt=paths=source_relative ./api/proto/macondo/macondo.proto`
