all: macondo_shell macondo_bot bot_shell make_gaddag make_leaves_structure

proto:
	protoc --go_out=gen --go_opt=paths=source_relative ./api/proto/macondo/macondo.proto

macondo_shell:
	go build -o bin/shell cmd/shell/main.go

macondo_bot:
	go build -o bin/bot cmd/bot/main.go

bot_shell:
	go build -o bin/bot_shell cmd/bot_shell/main.go

make_gaddag:
	go build -o bin/make_gaddag cmd/make_gaddag/main.go 

make_leaves_structure:
	go build -o bin/make_leaves_structure cmd/make_leaves_structure/main.go 

clean:
	rm -f bin/*
