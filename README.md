# macondo

A crossword game move generator, written in Go

Macondo is more than just a simple crossword game move generator! It will have a web interface and later simming capabilities and more.

# protoc

To generate pb files, run in this directory:

`protoc --twirp_out=gen --go_out=gen --go_opt=paths=source_relative ./api/proto/macondo/macondo.proto`
