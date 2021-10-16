import argparse
import timeit
from collections import deque
from heapq import heappush, heappop, heapify
import itertools
import os


class State:

    def __init__(self, state, parent, move, depth, cost, key):

        self.state = state

        self.parent = parent

        self.move = move

        self.depth = depth

        self.cost = cost

        self.key = key

        if self.state:
            self.map = ''.join(str(e) for e in self.state)

    def __eq__(self, other):
        return self.map == other.map

    def __lt__(self, other):
        return self.map < other.map

goal_state = [0, 1, 2, 3, 4, 5, 6, 7, 8]
goal_node = State
initial_state = list()
board_len = 0
board_side = 0

nodes_expanded = 0
max_search_depth = 0
max_frontier_size = 0

moves = list()
costs = set()


def move(state, position):

    new_state = state[:]

    index = new_state.index(0)

    if position == 1:  # Up

        if index not in range(0, board_side):

            temp = new_state[index - board_side]
            new_state[index - board_side] = new_state[index]
            new_state[index] = temp

            return new_state
        else:
            return None

    if position == 2:  # Down

        if index not in range(board_len - board_side, board_len):

            temp = new_state[index + board_side]
            new_state[index + board_side] = new_state[index]
            new_state[index] = temp

            return new_state
        else:
            return None

    if position == 3:  # Left

        if index not in range(0, board_len, board_side):

            temp = new_state[index - 1]
            new_state[index - 1] = new_state[index]
            new_state[index] = temp

            return new_state
        else:
            return None

    if position == 4:  # Right

        if index not in range(board_side - 1, board_len, board_side):

            temp = new_state[index + 1]
            new_state[index + 1] = new_state[index]
            new_state[index] = temp

            return new_state
        else:
            return None

def expand(node):

    global nodes_expanded
    nodes_expanded += 1

    neighbors = list()

    neighbors.append(State(move(node.state, 1), node, 1, node.depth + 1, node.cost + 1, 0))
    neighbors.append(State(move(node.state, 2), node, 2, node.depth + 1, node.cost + 1, 0))
    neighbors.append(State(move(node.state, 3), node, 3, node.depth + 1, node.cost + 1, 0))
    neighbors.append(State(move(node.state, 4), node, 4, node.depth + 1, node.cost + 1, 0))

    nodes = [neighbor for neighbor in neighbors if neighbor.state]

    return nodes

def ast(start_state):

    global max_frontier_size, goal_node, max_search_depth

    explored, heap, heap_entry, counter = set(), list(), {}, itertools.count()

    key = h1(start_state)

    root = State(start_state, None, None, 0, 0, key)

    entry = (key, 0, root)

    heappush(heap, entry)

    heap_entry[root.map] = entry

    while heap:

        node = heappop(heap)

        explored.add(node[2].map)

        if node[2].state == goal_state:
            goal_node = node[2]
            return heap

        neighbors = expand(node[2])

        for neighbor in neighbors:

            neighbor.key = neighbor.cost + h1(neighbor.state)

            entry = (neighbor.key, neighbor.move, neighbor)

            if neighbor.map not in explored:

                heappush(heap, entry)

                explored.add(neighbor.map)

                heap_entry[neighbor.map] = entry

                if neighbor.depth > max_search_depth:
                    max_search_depth += 1

            elif neighbor.map in heap_entry and neighbor.key < heap_entry[neighbor.map][2].key:

                hindex = heap.index((heap_entry[neighbor.map][2].key,
                                     heap_entry[neighbor.map][2].move,
                                     heap_entry[neighbor.map][2]))

                heap[int(hindex)] = entry

                heap_entry[neighbor.map] = entry

                heapify(heap)

        if len(heap) > max_frontier_size:
            max_frontier_size = len(heap)

def h1(state):

    return sum(abs(b % board_side - g % board_side) + abs(b//board_side - g//board_side)
               for b, g in ((state.index(i), goal_state.index(i)) for i in range(1, board_len)))

def read(configuration):

    global board_len, board_side

    data = configuration.split(",")

    for element in data:
        initial_state.append(int(element))

    board_len = len(initial_state)

    board_side = int(board_len ** 0.5)


def export(frontier, time):

    global moves

    moves = backtrace()

    file = open('output.txt', 'w')
    file.write("path_to_goal: " + str(moves))
    file.write("\ncost_of_path: " + str(len(moves)))
    file.write("\nnodes_expanded: " + str(nodes_expanded))
    file.write("\nfringe_size: " + str(len(frontier)))
    file.write("\nmax_fringe_size: " + str(max_frontier_size))
    file.write("\nsearch_depth: " + str(goal_node.depth))
    file.write("\nmax_search_depth: " + str(max_search_depth))
    file.write("\nrunning_time: " + format(time, '.8f'))
    file.close()

def backtrace():

    current_node = goal_node

    while initial_state != current_node.state:

        if current_node.move == 1:
            movement = 'U'
        elif current_node.move == 2:
            movement = 'D'
        elif current_node.move == 3:
            movement = 'L'
        else:
            movement = 'R'

        moves.insert(0, movement)
        current_node = current_node.parent

    return moves


def main():

    #parser = argparse.ArgumentParser()

    #parser.add_argument('algorithm')
    #parser.add_argument('board')
    #args = parser.parse_args()

    read("1,0,2,3,4,5,6,7,8")

    #function = function_map[args.algorithm]

    start = timeit.default_timer()

    frontier = ast(initial_state)

    stop = timeit.default_timer()

    export(frontier, stop-start)


if (__name__ == '__main__'):
    main()



