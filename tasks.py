# A binary builder for Macondo

from invoke import task


@task
def build(c, nickname):
    """Builds the shell for this machine's platform into
    macondo-<tag>-<nickname>.zip. Macondo links MAGPIE through cgo, so it is
    built natively (see .github/workflows/build-and-upload-binaries.yml);
    MAGPIE_MARCH picks the instruction set MAGPIE is compiled for."""
    tag = c.run("git describe --exact-match --tags", hide=True).stdout.strip()
    print("Tag was", tag)

    gitflags = f'-ldflags "-X main.GitVersion={tag}"'
    c.run("scripts/build_magpie.sh")
    c.run(f"CGO_ENABLED=1 go build -o macondo {gitflags} ./cmd/shell")
    c.run(f"zip -r macondo-{tag}-{nickname}.zip ./macondo ./data ./scripts/lua ./explainer")
    c.run("rm -f macondo")
