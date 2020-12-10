everything: all wasm

all: macondo_shell macondo_bot bot_shell gaddag_maker make_leaves_structure analyze

.PHONY: wasm

proto:
	protoc --go_out=gen --go_opt=paths=source_relative ./api/proto/macondo/macondo.proto

analyze:
	go build -trimpath -o bin/analyze cmd/analyze/main.go

macondo_shell:
	go build -trimpath -o bin/shell cmd/shell/main.go

macondo_bot:
	go build -trimpath -o bin/bot cmd/bot/main.go

bot_shell:
	go build -trimpath -o bin/bot_shell cmd/bot_shell/main.go

gaddag_maker:
	go build -trimpath -o bin/make_gaddag cmd/make_gaddag/main.go

make_leaves_structure:
	go build -trimpath -o bin/make_leaves_structure cmd/make_leaves_structure/main.go

wasm:
	GOOS=js GOARCH=wasm go build -trimpath -o ../liwords/liwords-ui/public/wasm/macondo.wasm wasm/*.go

clean:
	rm -f bin/*
