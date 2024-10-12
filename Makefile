everything: all wasm

all: macondo_shell macondo_ucgi macondo_bot bot_shell analyze

.PHONY: wasm

build-MacondoLambdaFunction:
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o bootstrap cmd/lambda/main.go
	cp ./bootstrap $(ARTIFACTS_DIR)/.

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

macondo_ucgi:
	go build -trimpath -o bin/ucgi_cli cmd/ucgi_cli/main.go

# wasm:
# 	GOOS=js GOARCH=wasm go build -trimpath -o ../liwords/liwords-ui/public/wasm/macondo.wasm wasm/*.go

clean:
	rm -f bin/*
