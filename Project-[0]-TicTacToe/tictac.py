#!/usr/bin/python3
# Author: Rino David

import numpy as np
import argparse


class TicTacToe:
    def __init__(self, board=None, player=1) -> None:
        if board is None:
            self.board = self.init_board()
        else:
            self.board = board
        self.player = player

    def init_board(self):
        return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    def print_board(self):
        print(self.board)

    def eval_win(self):
        # horizontal case
        openspaces = 0
        for x in range(3):
            for y in range(3):
                if self.board[x][y] == 0:
                    openspaces += 1

        if openspaces == 0:
            return 0

        for x in range(3):
            if check3(self.board[x][0], self.board[x][1], self.board[x][2]):
                return self.board[x][0]
        # Vertical case
        for x in range(3):
            if check3(self.board[0][x], self.board[1][x], self.board[2][x]):
                return self.board[0][x]
        # Diagonal Case
        # Right Diagonal
        if check3(self.board[0][0], self.board[1][1], self.board[2][2]):
            return self.board[0][0]
        elif check3(self.board[2][0], self.board[1][1], self.board[0][2]):
            return self.board[2][0]

        # None true
        return None
    def play_game(self):

        bestMove = None
        instantWin = None
        tempBoard = [0,0]

        while self.eval_win() is None:
            tempBoard = self.board
            instantWin, bestMove = checkInstantWin(self)
            if instantWin == False:
                bestMove = minimax(self)
            self.board = tempBoard
            self.board[bestMove[0], bestMove[1]] = self.player
            print("Player: " + str(self.player) + " turn")
            self.player = -self.player
            print(self.board)

        return self.board, self.eval_win()

def load_board(filename):
    return np.loadtxt(filename)


# Utility Functions

def minimax(self):
    tempValue, bestMove = max_value(self, 0)
    print(tempValue)
    return bestMove


def check3(a, b, c):
    return a == b and b == c and a != 0


def max_value(self, depth):
    result = self.eval_win()

    if result == 1:
        return 10, None
    elif result == -1:
        return -10, None
    elif result == 0:
        return 0, None


    bestScore = float('-inf')
    bestMove = [0, 0]
    for x in range(3):
        for y in range(3):
            if self.board[x][y] == 0:
                self.board[x][y] = self.player
                #print("Max")
                #print(self.board)
                score, tempMove = min_value(self, depth + 1)
                self.board[x][y] = 0
                score -= depth
                if score > bestScore:
                    bestScore = score
                    bestMove = [x, y]


                #print("Max")
                #print(self.board)


    return bestScore, bestMove

def checkInstantWin(self):
    bestMove = None
    for x in range (3):
        for y in range (3):
            if self.board[x][y] == 0:
                self.board[x][y] = -self.player
                winvalue = self.eval_win()
                if winvalue == 1 or winvalue == -1:
                    self.board[x][y] = 0
                    bestMove = [x, y]
                    return True, bestMove
                self.board[x][y] = 0

    return False, None

def min_value(self, depth):
    result = self.eval_win()

    if result == 1:
        return 10, None
    elif result == -1:
        return -10, None
    elif result == 0:
        return 0, None

    bestScore = float('inf')
    bestMove = [0, 0]

    for x in range(3):
        for y in range(3):
            if self.board[x][y] == 0:
                self.board[x][y] = -self.player
                #print("min")
                #print(self.board)
                score, tempMove = max_value(self, depth + 1)
                self.board[x][y] = 0
                score += depth
                if score < bestScore:
                    bestScore = score
                    bestMove = [x, y]
                #print("min")
                #print(self.board)
    return bestScore, bestMove





# def save_board( self, filename ):
# 	np.savetxt( filename, self.board, fmt='%d')

def main():
    parser = argparse.ArgumentParser(description='Play tic tac toe')
    parser.add_argument('-f', '--file', default=None, type=str, help='load board from file')
    parser.add_argument('-p', '--player', default=1, type=int, choices=[1, -1],
                        help='player that playes first, 1 or -1')
    args = parser.parse_args()

    board = load_board(args.file) if args.file else None
    testcase = np.array([[0, 0, 0],
                         [-1, 1, 0],
                         [-1, 0, 0]])
    ttt = TicTacToe(testcase, args.player)
    # ttt.print_board()
    b, p = ttt.play_game()
    print("final board: \n{}".format(b))
    print("winner: player {}".format(p))


if __name__ == '__main__':
    main()