all: macondo_shell macondo_bot bot_shell mlproducer analyzer_worker

.PHONY: magpie

# MAGPIE's static libraries (third_party/magpie, a submodule), which the
# magpie/ cgo package links. MAGPIE_MARCH=x86-64-v2 (or similar) for a
# portable build.
magpie:
	scripts/build_magpie.sh

# Macondo links MAGPIE through cgo. The Lambda runs on Amazon Linux 2, so the
# binary is linked statically (build it with `sam build --use-container` so
# it is compiled there), with MAGPIE for a portable instruction set.
build-MacondoLambdaFunction:
	MAGPIE_MARCH=x86-64-v2 scripts/build_magpie.sh
	GOOS=linux GOARCH=amd64 CGO_ENABLED=1 go build -tags netgo,osusergo -ldflags '-linkmode external -extldflags "-static"' -o bootstrap cmd/lambda/main.go
	cp ./bootstrap $(ARTIFACTS_DIR)/.

proto:
	protoc --go_out=gen --go_opt=paths=source_relative ./api/proto/macondo/macondo.proto

macondo_shell: magpie
	go build -trimpath -ldflags "-X main.GitVersion=$(shell git describe --tags --always)" -o bin/shell cmd/shell/main.go

macondo_bot: magpie
	go build -trimpath -o bin/bot cmd/bot/main.go

bot_shell: magpie
	go build -trimpath -o bin/bot_shell cmd/bot_shell/main.go

mlproducer: magpie
	go build -trimpath -o bin/mlproducer cmd/mlproducer/*.go

analyzer_worker: magpie
	go build -trimpath -ldflags "-X main.GitVersion=$(shell git describe --tags --always)" -o bin/analyzer-worker cmd/analyzer-worker/main.go

inferdiag: magpie
	go build -trimpath -o bin/inferdiag cmd/inferdiag/main.go

taufit: magpie
	go build -trimpath -o bin/taufit cmd/taufit/main.go

clean:
	rm -f bin/*
	rm -rf build/magpie
