FROM golang:alpine
ENV GOPATH=/go
ADD . /go/src/github.com/domino14/macondo
WORKDIR /go/src/github.com/domino14/macondo
RUN go build

EXPOSE 8088
CMD ./macondo -dawgpath=/dawgs/