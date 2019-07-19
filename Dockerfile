FROM golang:alpine
ENV GOPATH=/go
ADD . /opt/macondo
WORKDIR /opt/macondo
RUN go build

EXPOSE 8088
CMD ./macondo