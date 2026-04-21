everything: all wasm

all: macondo_shell macondo_bot bot_shell mlproducer analyzer_worker macondo_notebook

.PHONY: wasm

build-MacondoLambdaFunction:
	GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o bootstrap cmd/lambda/main.go
	cp ./bootstrap $(ARTIFACTS_DIR)/.

proto:
	protoc --go_out=gen --go_opt=paths=source_relative ./api/proto/macondo/macondo.proto

macondo_shell:
	go build -trimpath -ldflags "-X main.GitVersion=$(shell git describe --tags --always)" -o bin/shell cmd/shell/main.go

macondo_bot:
	go build -trimpath -o bin/bot cmd/bot/main.go

bot_shell:
	go build -trimpath -o bin/bot_shell cmd/bot_shell/main.go

mlproducer:
	go build -trimpath -o bin/mlproducer cmd/mlproducer/*.go

analyzer_worker:
	go build -trimpath -ldflags "-X main.GitVersion=$(shell git describe --tags --always)" -o bin/analyzer-worker cmd/analyzer-worker/main.go

macondo_notebook:
	go build -trimpath -ldflags "-X main.GitVersion=$(shell git describe --tags --always)" -o bin/macondo-notebook cmd/notebook/main.go

# wasm:
# 	GOOS=js GOARCH=wasm go build -trimpath -o ../liwords/liwords-ui/public/wasm/macondo.wasm wasm/*.go

clean:
	rm -f bin/*
