package alphabeta

import (
	"fmt"
	"io/ioutil"
)

// Attempt to visualize minimax graph with dot

type dotfile struct {
	declarations []string
	directives   []string
}

func genDotFile(n *GameNode, d *dotfile) {
	if len(n.children) == 0 {
		// terminal node
		// decl := fmt.Sprintf("%p [label=\"%v\nV: %v\"];",
		// 	n, joinDesc(n.move.ShortDescription()), n.move.Valuation(),
		// )
		// declarations = append(declarations, decl)
		return
	}

	parent := n
	for _, child := range parent.children {
		decl := fmt.Sprintf("n_%p [label=\"%v\\nPlayVal: %v\\nNodeVal: %v\"];",
			child, child.move.ShortDescription(),
			child.move.Valuation(), child.heuristicValue)

		conn := fmt.Sprintf("n_%p -> n_%p;", parent, child)
		d.declarations = append(d.declarations, decl)
		d.directives = append(d.directives, conn)
		genDotFile(child, d)
	}
}

func saveDotFile(root *GameNode, d *dotfile, outFile string) {
	out := ""
	out += fmt.Sprintf("digraph {\n")
	out += fmt.Sprintf(" n_%p [label=\"(root)\"]\n", root)
	for _, d := range d.declarations {
		out += fmt.Sprintf(" %v\n", d)
	}
	out += fmt.Sprintf("\n")
	for _, d := range d.directives {
		out += fmt.Sprintf(" %v\n", d)
	}
	out += fmt.Sprint("}\n")
	ioutil.WriteFile(outFile, []byte(out), 0644)
}
