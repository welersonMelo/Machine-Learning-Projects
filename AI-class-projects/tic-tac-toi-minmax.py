#min max to tic tac toi jogo da velha

import os

tabuleiroP = ['123','456','789']

def telaJogo():
	global tabuleiroP
	print(tabuleiroP[0][0],'|',tabuleiroP[0][1],'|',tabuleiroP[0][2])
	print(tabuleiroP[1][0],'|',tabuleiroP[1][1],'|',tabuleiroP[1][2])
	print(tabuleiroP[2][0],'|',tabuleiroP[2][1],'|',tabuleiroP[2][2])
	
def ganhou(jogador, tabuleiro):
	if jogador == 1:
		padrao = 'XXX'
	else:
		padrao = 'OOO'
	if tabuleiro[0] == padrao or tabuleiro[1] == padrao or tabuleiro[2] == padrao:
		return True
	t = []
	t.append(tabuleiro[0][0]+tabuleiro[1][0]+tabuleiro[2][0])
	t.append(tabuleiro[0][1]+tabuleiro[1][1]+tabuleiro[2][1])
	t.append(tabuleiro[0][2]+tabuleiro[1][2]+tabuleiro[2][2])
	
	if t[0] == padrao or t[1] == padrao or t[2] == padrao:
		return True
	
	d1 = t[0][0]+t[1][1]+t[2][2]
	d2 = t[0][2]+t[1][1]+t[2][0]
	if d1 == padrao or d2 == padrao:
		return True
		
	return False

def velha(tabuleiro):
	t = tabuleiro
	if t[0][0]!='1' and t[0][1]!='2' and t[0][2]!='3' and t[1][0]!='4' and t[1][1]!='5' and t[1][2]!='6' and t[2][0]!='7' and t[2][1]!='8' and t[2][2]!='9':
		return True
	return False

def jogadaAt(jogada, jogador, tabuleiro):
	tabuleiro[0] = tabuleiro[0].replace(jogada, jogador)
	tabuleiro[1] = tabuleiro[1].replace(jogada, jogador)
	tabuleiro[2] = tabuleiro[2].replace(jogada, jogador)
	return tabuleiro
	
def jogadaUser(tabuleiro):
	jogada = input('Sua jogada: ')
	return jogadaAt(jogada, 'X', tabuleiro)

def actionState(tabuleiro):
	actions = []
	for line in tabuleiro:
		for s in line:
			if(s != 'X' and s != 'O'):
				actions.append(s)
	
	return actions

def minValue(tabuleiro):
	if ganhou(1, tabuleiro):
		return -1
	if ganhou(2, tabuleiro):
		return 1
	if velha(tabuleiro):
		return 0
	
	actions = actionState(tabuleiro)
	v = 10
	
	for a in actions:
		newTab = jogadaAt(a, 'X', tabuleiro[:])
		v = min(v, maxValue(newTab[:]))
	return v
	
def maxValue(tabuleiro):
	if ganhou(1, tabuleiro):
		return -1
	if ganhou(2, tabuleiro):
		return 1
	if velha(tabuleiro):
		return 0
	
	actions = actionState(tabuleiro)
	v = -10
	
	for a in actions:
		newTab = jogadaAt(a, 'O', tabuleiro[:])
		v = max(v, minValue(newTab[:]))
	return v
	
def jogadaMaquina(tabuleiro):
	actions = actionState(tabuleiro)
	ans = -1
	maior = -10
	for a in actions:
		newTab = jogadaAt(a, 'O', tabuleiro[:])
		v = minValue(newTab)
		if v > maior:
			maior = v
			ans = a
	
	jogada = ans
	return jogadaAt(jogada, 'O', tabuleiro[:])	

####### Main ########
print ('Welcome!\nDigite um dos números para jogar na casa desejada!')
telaJogo()

while True:
	jogadaUser(tabuleiroP)
	os.system('clear')
	telaJogo()
	if ganhou(1, tabuleiroP):
		print ('Jogador 1, você, ganhou!')
		break
		
	if velha(tabuleiroP):
		print ('Deu velha!')
		break
	
	tabuleiroP = jogadaMaquina(tabuleiroP[:])
	os.system('clear')
	telaJogo()
	if ganhou(2, tabuleiroP):
		print ('Jogador 2, IA, ganhou!')
		break
