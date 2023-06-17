import sys
import yaml

def process(filename):
    f = open(filename)
    log = yaml.safe_load(f)
    f.close()

    print("loaded plies:", len(log))
    cursor = log
    last_cursor = cursor
    while True:
        line = input("> ")
        if line.startswith("help"):
            print("Usage: ")
            print("l - list current level of tree")
            print("s{number} - go down level of tree")
            print("b - go back where you were")
            print("r - reset back to top of tree")
            print("p - don't use this")
        if line.startswith("l"):
            # list
            for idx, m in enumerate(cursor):
                if "ply" in m:
                    print(f'{idx}) ply: {m["ply"]}, plays: {len(m["plays"])}')
                elif "play" in m:                    
                    print(f'{idx} play: {m["play"]} ({m["value"]})')
        elif line.startswith("s"):
            try:
                n = int(line[1:])
            except ValueError:
                print("?")
                continue
            try:
                placeholder = cursor
                cursor = cursor[n]["plays"]
                last_cursor = placeholder
            except IndexError:
                print("IndexError, try again")
            except KeyError:
                print("You're at the bottom of the tree.")
        elif line.startswith("b"):
            cursor = last_cursor
        elif line.startswith("r"):
            last_cursor = cursor
            cursor = log
        elif line.startswith("p"):
            print(cursor)
if __name__ == '__main__':
    process(sys.argv[1])