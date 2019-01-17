// Package endgame solves the endgame. It uses some sort of efficient tree
// search.
package endgame

import (
	"sort"

	"github.com/domino14/macondo/alphabet"
	"github.com/domino14/macondo/gaddag"

	"github.com/domino14/macondo/board"

	"github.com/domino14/macondo/move"
	"github.com/domino14/macondo/movegen"
)

// I guess Infinity isn't that big.
const Infinity = 1e6

const CostConstant = 5000

type TreeNode struct {
	move     *move.Move
	children []*TreeNode
}

func newChildrenNodes(children []*move.Move) []*TreeNode {
	// sort children by descending score. We'll basically always want
	// to do this.
	sort.Slice(children, func(i, j int) bool {
		return children[i].Score() > children[j].Score()
	})

	treechildren := make([]*TreeNode, len(children))
	for idx := range children {
		treechildren[idx] = &TreeNode{children[idx], nil}
	}
	return treechildren
}

func newTreeNode(move *move.Move, children []*move.Move) *TreeNode {
	return &TreeNode{move, newChildrenNodes(children)}
}

// AStarSolver implements the A* algorithm. I don't know how to implement
// B* so let's see how this does first, and improve on it later. Maybe
// I'll learn something.
type AStarSolver struct {
	movegen          *movegen.GordonGenerator
	playerOnTurnRack *alphabet.Rack
	opponentRack     *alphabet.Rack
	board            *board.GameBoard
	gd               *gaddag.SimpleGaddag
	bag              *alphabet.Bag
}

func (a *AStarSolver) Init(board *board.GameBoard, movegen *movegen.GordonGenerator,
	bag *alphabet.Bag, gd *gaddag.SimpleGaddag) {

	a.board = board
	a.movegen = movegen
	a.bag = bag
	a.gd = gd
}

// Solve returns the best move.
// func (a *AStarSolver) Solve(board *board.GameBoard, playerOnTurn, opponent *alphabet.Rack,
// 	movegen *movegen.GordonGenerator) *move.Move {

// 	// Start by building the tree.
// 	movegen.GenAll(playerOnTurn)
// 	startNode := newTreeNode(nil, movegen.Plays())

// 	// A lot of this comes from Wikipedia
// 	closedSet := map[*TreeNode]bool(nil)
// 	openSet := map[*TreeNode]bool{startNode: true}
// 	cameFrom := map[*TreeNode]*TreeNode(nil)

// 	// For each node, the cost of getting from the start node to that node
// 	// (default to infinity).
// 	gScore := map[*TreeNode]int{startNode: 0}
// 	// For each node, the total cost of getting from the start node to the goal
// 	// by passing by that node. That value is partly known, partly heuristic.
// 	fScore := map[*TreeNode]int{startNode: a.heuristicEstimate(startNode, opponent)}
// 	return nil
// }

func (a *AStarSolver) heuristicEstimate(node *TreeNode, opponentRack *alphabet.Rack) int {
	// Calculate a heuristic estimate for the cost from the node to any goal.
	// A goal is just a finished endgame. Obviously we want the best one.
	// Define cost as CostConstant - spread. That way it can be minimized.

	firstMoveScore := node.children[0].move.Score()
	// A decent spread heuristic can be obtained from the opponent's
	// highest-score reply to this move. ¯\_(ツ)_/¯
	a.board.PlayMove(node.children[0].move, a.gd, a.bag, true)
	a.movegen.GenAll(opponentRack)
	node.children[0].children = newChildrenNodes(a.movegen.Plays())
	replyMoveScore := node.children[0].children[0].move.Score()
	a.board.RestoreFromBackup()
	return CostConstant - (firstMoveScore - replyMoveScore)
}
