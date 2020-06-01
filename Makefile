all: macondo_shell make_gaddag make_leaves_structure

# macondo_shell cannot currently be written to bin/ because it looks for some
# datafiles in paths relative to the location of the executable.
macondo_shell:
	go build -o macondo_shell main.go

make_gaddag:
	go build -o bin/make_gaddag cmd/make_gaddag/main.go 

make_leaves_structure:
	go build -o bin/make_leaves_structure cmd/make_leaves_structure/main.go 

clean:
	rm -f macondo_shell bin/*
