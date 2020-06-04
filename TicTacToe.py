# -*- coding: utf8 -*-

# ***** This is a stub for tic-tac-toe *****
# may be useful for future applications

# TO-DO:

# Done:

import random
import sys
import math
import os

import GUI

# ***** For Tic Tac Toe:
board = [ [' ']*3 for i in range(3)]

def hasWinner():
	global board

	for player in ['X', 'O']:
		tile = player

		# check horizontal
		for i in [0, 1, 2]:
			if board[i][0] == tile and board[i][1] == tile and board[i][2] == tile:
				return player

		# check vertical
		for j in [0, 1, 2]:
			if board[0][j] == tile and board[1][j] == tile and board[2][j] == tile:
				return player

		# check diagonal
		if board[0][0] == tile and board[1][1] == tile and board[2][2] == tile:
			return player

		# check backward diagonal
		if board[0][2] == tile and board[1][1] == tile and board[2][0] == tile:
			return player

	# ' ' is for game still open
	for i in [0, 1, 2]:
		for j in [0, 1, 2]:
			if board[i][j] == ' ':
				return ' '

	# '-' is for draw match
	return '-'

def opponentPlay():
	global board

	playable2 = []
	for i in [0, 1, 2]:
		for j in [0, 1, 2]:
			if board[i][j] == ' ':
				playable2.append((i,j))
	return random.choice(playable2)

def printBoard():
	global board
	for i in [0, 1, 2]:
		print(' [', end='')
		for j in [0, 1, 2]:
			print(board[i][j], end='')
		print(']')

# TO-DO: actions could be intermediate predicates
def playGames():
	from GUI import draw_board
	global board
	win = draw = stall = lose = 0

	for n in range(1000):		# play game N times
		print("\t\tGame ", n, end='\r')
		# Initialize board
		for i in [0, 1, 2]:
			for j in [0, 1, 2]:
				if board[i][j] != ' ':
					rete_net.remove_wme(WME(board[i][i], str(i), str(j)))
				rete_net.add_wme(WME(' ', str(i), str(j)))
				board[i][j] = ' '

		CurrentPlayer = 'X'					# In the future, may play against self
		moves = []							# for recording played moves
		for move in range(9):				# Repeat playing moves in single game
			# print("    move", move, end='; ')

			if CurrentPlayer == 'X':
				board[x][y] = CurrentPlayer
				moves.append(candidate)

			else:			# Player = 'O'
				i,j = opponentPlay()
				board[i][j] = 'O'

			printBoard()		# this is text mode
			draw_board(board)	# graphics mode

			# check if win / lose, assign rewards accordingly
			winner = hasWinner()
			if winner == ' ':
				# let the same set of rules play again
				# let opponent play (opponent = self? this may be implemented later)
				CurrentPlayer = 'O' if CurrentPlayer == 'X' else 'X'
			elif winner == '-':
				# increase the scores of all played moves by 3.0
				for candidate in moves:
					candidate['fitness'] += 3.0
				# print("Draw")
				draw += 1
				break			# next game
			elif winner == 'X':
				# increase the scores of all played moves by 10.0
				for candidate in moves:
					candidate['fitness'] += 10.0
				# print("X wins")
				win += 1
				break			# next game
			elif winner == 'O':
				# decrease the scores of all played moves by 8.0
				for candidate in moves:
					candidate['fitness'] -= 8.0
				# print("O wins")
				lose += 1
				break			# next game
	return win, draw, stall, lose

print("\n\x1b[32m——`—,—{\x1b[31;1m@\x1b[0m\n")   # Genifer logo ——`—,—{@

print("\x1b[36m**** This program works till here....\n\x1b[0m")
os.system("beep -f 2000 -l 50")
exit(0)
