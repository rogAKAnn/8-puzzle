import argparse
import random
import numpy as np
import timeit
from collections import deque
from heapq import heappush, heappop, heapify
import itertools
import os
from State import State


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

def init():
    global goal_state, goal_node, initial_state, board_len, board_side, nodes_expanded, max_search_depth, max_frontier_size, moves, costs

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

    return nodes #List of state of next move

def ast(start_state, h):

    global max_frontier_size, goal_node, max_search_depth

    explored, heap, heap_entry, counter = set(), list(), {}, itertools.count()

    key = h(start_state)

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

            neighbor.key = neighbor.cost + h(neighbor.state)

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

def h1(state): #mahattan

    return sum(abs(b % board_side - g % board_side) + abs(b//board_side - g//board_side)
               for b, g in ((state.index(i), goal_state.index(i)) for i in range(1, board_len)))

def h2(state): #misplaced tile
    return sum([0 if b==g else 1 for b,g in ((state.index(i),goal_state.index(i)) for i in range(1, board_len))])

def h3(state): #Gashnig
    count = 0
    temp = state + []
    while temp != goal_state:
        if temp.index(0) == goal_state.index(0):
            for i in range(1,board_len):
                if temp.index(i) != goal_state.index(i):
                    index0 = temp.index(0)
                    indexMatchedTile = temp.index(i)
                    temp[index0] = i
                    temp[indexMatchedTile] = 0
                    break
        else:
            index0 = temp.index(0) # 0 dang tai vi o day
            MatchedTile = goal_state[index0] # Vi tri dang le o do
            temp[temp.index(MatchedTile)] = 0
            temp[index0] = MatchedTile
        count += 1
    return count


     

def read(configuration):

    global board_len, board_side, initial_state

    initial_state = configuration

    board_len = len(initial_state)

    board_side = int(board_len ** 0.5)


def export(frontier, time, n,ouput1):

    global moves, o

    moves = backtrace()

    file = open(ouput1, 'a')

    m = [len(moves)] + [nodes_expanded] + [len(frontier)] + [max_frontier_size] + [goal_node.depth] + [max_search_depth] + [time]
    o += [m] 

    file.write('|' +format(str(n), "^5"))
    file.write('|' +format(str(initial_state), "^30"))

    file.write('|' +format(str(len(moves)), "^15"))
    file.write('|' +format(str(nodes_expanded), "^15" ))
    file.write('|' +format(str(len(frontier)), "^15" ))
    file.write('|' +format(str(max_frontier_size), "^20" ))
    file.write('|' +format(str(goal_node.depth), "^15" ))
    file.write('|' +format(str(max_search_depth), "^18" ))
    file.write('|' +format(time, '^20.8f')+'|'  + '\n')

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

def isSolvable(state):
    invCount = 0
    for i in range(2, len(state)):
        for j in range (1, i):
            if state.index(i) < state.index(j):
                invCount = invCount + 1
    return invCount % 2 == 0
        

def export_(input, output):
    global o

    o = []

    file = open(output, 'w')

    file.write(('h1: Manhattan' if output == 'output_h1.txt' else 'h2: Misplaced Tiles' if output=='output_h2.txt' else 'h3: Gashnig') + '\n')
    file.write('|' +format('ID', "^5"))
    file.write('|' +format('input', "^30"))

    file.write('|' +format('cost of path', "^15"))
    file.write('|' +format('nodes expanded', "^15" ))
    file.write('|' +format('fringe size', "^15" ))
    file.write('|' +format('max frontier size', "^20" ))
    file.write('|' +format('search depth', "^15" ))
    file.write('|' +format('max search depth', "^18" ))
    file.write('|' +format('running time', '^20')+'|'  + '\n')

    file.close()


    count1 = 0
    for x in input:
            init()
            read(x)
            start = timeit.default_timer()
            frontier = ast(x, h1 if output == 'output_h1.txt' else h2 if output == 'output_h2.txt' else h3)
            stop = timeit.default_timer()
            export(frontier, stop-start, count1 + 1, output)
            count1 += 1

    o1 = np.average(o,0)

    file = open(output, 'a')

    file.write( '\n|' +format('Average', "^36"))

    file.write('|' +format(str(o1[0]), "^15"))
    file.write('|' +format(str(o1[1]), "^15" ))
    file.write('|' +format(str(o1[2]), "^15" ))
    file.write('|' +format(str(o1[3]), "^20" ))
    file.write('|' +format(str(o1[4]), "^15" ))
    file.write('|' +format(str(o1[5]), "^18" ))
    file.write('|' +format(o1[6], '^20.8f')  + '|\n' )

    file.close()


def printBoard(state):
    global board_side
    for i in range(0, board_len, board_side):
        print(state[i],end='')
        for j in range(i+1,i+board_side):
            print(' |', state[j], end='')
        print()
    print("---------")

o = []
            
def main():
    
    test_number = 10
    count = 0
    input = []
    while count != test_number:
        init()
        initial_state = goal_state + []
        random.shuffle(initial_state)     
        if(isSolvable(initial_state)):
            input += [initial_state]
            count += 1

    output1 = 'output_h1.txt'
    output2 = 'output_h2.txt'
    output3 = 'output_h3.txt'

    export_(input, output1)
    export_(input, output2)
    export_(input, output3)



if (__name__ == '__main__'):
    main()



