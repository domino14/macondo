FROM golang:alpine
RUN apk add --no-cache bash build-base
ENV GOPATH=/go
ADD . /opt/macondo
WORKDIR /opt/macondo
RUN MAGPIE_MARCH=x86-64-v2 scripts/build_magpie.sh && go build ./...

EXPOSE 8088
CMD ./macondo