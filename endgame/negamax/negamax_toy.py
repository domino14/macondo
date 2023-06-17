# An extremely simple toy example to help me visualize this.

class Node:
    def __init__(self, board, rack1, rack2, score1, score2, onturn):
        self.board = board
        self.rack1 = rack1
        self.rack2 = rack2
        self.score1 = score1
        self.score2 = score2
        self.onturn = onturn
        self.children = []
    
    def __repr__(self):
        return f"<Node: Board {self.board}, Racks: {self.rack1} - {self.rack2}, Scores: {self.score1} - {self.score2}, Onturn: {self.onturn}>"
    
    def spread_for(self, player):
        spread = self.score1 - self.score2
        if player == 1:
            spread = -spread
        return spread


topState = Node("RADAR/     /     ", "ET", "ET", 35, 18, 0)
topState.children = [
    Node("RADAR/ T   /     ", "E", "ET", 39, 18, 1),
    Node("RADAR/   T /     ", "E", "ET", 37, 18, 1)]
# left side
topState.children[0].children = [
    Node("RADAR/ T T /     ", "E", "E", 39, 20, 0)]
topState.children[0].children[0].children = [
    Node("RADAR/ T T / E   ", "", "E", 44, 20, 1)]
# right side
topState.children[1].children = [
    Node("RADAR/ T T /     ", "E", "E", 37, 22, 0)]
topState.children[1].children[0].children = [
    Node("RADAR/ T T / E   ", "", "E", 42, 22, 1)]


def evaluate(node):
    onturn = node.onturn
    spreadNow = node.score1 - node.score2
    initialSpread = beginningSpread

    if onturn == 1:
        spreadNow = -spreadNow
        initialSpread = -initialSpread

    return spreadNow - initialSpread

# beginning spread is relative to left player (1)
beginningSpread = topState.score1 - topState.score2

def negamax(node, depth):
    if depth == 0 or len(node.children) == 0:
        evaluation = evaluate(node)
        print("  " * depth, "evaluation returned", evaluation)
        return evaluation
    value = -100000
    our_spread = node.spread_for(node.onturn)
    for child in node.children:
        cur_our_spread = child.spread_for(node.onturn)
        spread_change = cur_our_spread - our_spread

        negavalue = negamax(child, depth-1)
        print ("  " * depth, "- child", child, "negavalue", -negavalue, "spread change", spread_change)
        value = max(value, -negavalue)
    print("  " * depth, "node", node, "value returned for this node", value)
    print("  " * depth, "i would store", value )
    return value


if __name__ == '__main__':
    print("beginning state", topState)
    print(negamax(topState, depth=3))