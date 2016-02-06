FROM golang:latest
ENV GOPATH=/go
ADD . /go/src/github.com/domino14/macondo
WORKDIR /go/src/github.com/domino14/macondo
RUN go get
RUN go build