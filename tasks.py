# A binary builder for Macondo

from invoke import task

# Windows targets cross-compile from Linux with llvm-mingw (its bin directory
# on PATH), as MAGPIE's own Windows release does.
WINDOWS_TRIPLES = {"amd64": "x86_64-w64-mingw32", "arm64": "aarch64-w64-mingw32"}


@task
def build(c, nickname, windows=""):
    """Builds the shell into macondo-<tag>-<nickname>.zip. Macondo links MAGPIE
    through cgo, so it is built natively on each platform (see
    .github/workflows/build-and-upload-binaries.yml), except for Windows:
    --windows amd64 (or arm64) cross-compiles macondo.exe and updater.exe.
    MAGPIE_MARCH picks the instruction set MAGPIE is compiled for."""
    # A tag on release builds; a commit hash on branch test runs.
    tag = c.run("git describe --tags --always", hide=True).stdout.strip()
    print("Tag was", tag)

    if not windows:
        gitflags = f'-ldflags "-X main.GitVersion={tag}"'
        c.run("scripts/build_magpie.sh")
        c.run(f"CGO_ENABLED=1 go build -o macondo {gitflags} ./cmd/shell")
        c.run(
            f"zip -r macondo-{tag}-{nickname}.zip ./macondo ./data ./scripts/lua ./explainer"
        )
        c.run("rm -f macondo")
        return

    triple = WINDOWS_TRIPLES[windows]
    # Static, so the executables need no MinGW DLLs; an 8 MB stack for every
    # thread, which MAGPIE's recursion needs (see magpie/magpie.go).
    ldflags = (
        f'-ldflags "-X main.GitVersion={tag} '
        "-extldflags '-static -Wl,--stack,8388608'\""
    )
    env = f"GOOS=windows GOARCH={windows} CGO_ENABLED=1 CC={triple}-clang"
    c.run(f"MAGPIE_TARGET=windows-{windows} scripts/build_magpie.sh")
    c.run(f"{env} go build -o macondo.exe {ldflags} ./cmd/shell")
    c.run(f"{env} go build -o updater.exe {ldflags} ./cmd/updater")
    c.run(
        f"zip -r macondo-{tag}-{nickname}.zip ./macondo.exe ./updater.exe ./data ./scripts/lua ./explainer"
    )
    c.run("rm -f macondo.exe updater.exe")
