# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:40:52 2021

@author: Milena Jordão Rempto
"""

# automatizar a construcao de X,Y,IEN, cc e inner ✔

# implementar uma funcao para escolha de distribuicao de espacamento em x,y
#  - x (linear) ----> dx constante ✔
#  - x*x (quadratica) ✔
#  - x*x*x (cubica) ✔
#  - exp(x) (exponencial) ✔
#  - gaussiana(x) (gaussiana) ✔
# consertar IEN ✔
# implementar o mesmo problema para malha triangular (modificar a IEN) ✔
# colocar espaçamento independente pra x e y ✔
# implementar vizinho e vizinhoElem e plotar quantidade de vizinhos ✔



# implementar função que analisa se todos os elementos estão no sentido horário ou anti-horário 
# (produto vetorial de dois vetores consecutivos a partir da IEN positivo ou negativo)
# plotar distribuição da área do elementos (metade do determinante entre dois vetores = metade da área do paralelograma, OU fórmula de Heron)
# somar valores das áreas para encontrar a área do domínio
# criar malha com gmesh e importar (ver vídeo e slides da graduação)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
from matplotlib.ticker import MaxNLocator

###################################################################################################################################################
#inputs
Lx = 1.0
Ly = 1.0
nx = 4
ny = 4

#elemento = 'quad'
elemento = 'triangle'

espacamento_em_x = 'linear'
#espacamento_em_x = 'quadratica'
#espacamento_em_x = 'cubica'
#espacamento_em_x = 'exponencial'
#espacamento_em_x = 'gaussiana'

espacamento_em_y = 'linear'
#espacamento_em_y = 'quadratica'
#espacamento_em_y = 'cubica'
#espacamento_em_y = 'exponencial'
#espacamento_em_y = 'gaussiana'
################################################################################################################################################
npoints = nx*ny

if elemento == 'quad':
    ne = (nx-1)*(ny-1)
elif elemento == 'triangle': 
    ne = (nx-1)*(ny-1)*2


# construcao dos vetores X e Y

X_ = np.linspace(0,1,nx)
Y_ = np.linspace(0,1,ny)

if espacamento_em_x == 'linear':
    X_ = (X_**1)*Lx
    X=np.tile(X_, ny)
elif espacamento_em_x == 'quadratica':
    X_ = (X_**2)*Lx
    X=np.tile(X_, ny)
elif espacamento_em_x == 'cubica':
    X_ = (X_**3)*Lx
    X=np.tile(X_, ny)    
elif espacamento_em_x == 'exponencial':
    X_ = (np.exp(X_))*Lx
    X=np.tile(X_, ny)
elif espacamento_em_x == 'gaussiana':
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    mu = 1 #mean 
    sig = 0.5 #standard deviation
    X_ = (gaussian(X_, mu, sig))*Lx
    X=np.tile(X_, ny)

if espacamento_em_y == 'linear':
    Y_ = (Y_**1)*Ly
    Y=np.repeat(Y_, nx)
elif espacamento_em_y == 'quadratica':
    Y_ = (Y_**2)*Ly
    Y=np.repeat(Y_, nx)
elif espacamento_em_y == 'cubica':
    Y_ = (Y_**3)*Ly
    Y=np.repeat(Y_, nx)  
elif espacamento_em_y == 'exponencial':
    Y_ = (np.exp(Y_))*Ly
    Y=np.repeat(Y_, nx)
elif espacamento_em_y == 'gaussiana':
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    mu = 1 #mean 
    sig = 0.5 #standard deviation
    Y_ = (gaussian(Y_, mu, sig))*Ly
    Y=np.repeat(Y_, nx)



# geracao da matriz de conectividade IEN

IEN = []
if elemento == 'quad':
    for j in range (ny-1):
        for i in range(nx-1):
            IEN.append([i+j*nx, i+1+j*nx, i+nx+1+j*nx, i+nx+j*nx])
elif elemento == 'triangle':
    for j in range (ny-1):
        for i in range(nx-1):
            IEN.append([i+j*nx, i+nx+1+j*nx, i+nx+j*nx])
            IEN.append([i+j*nx, i+1+j*nx, i+nx+1+j*nx])
        
IEN = np.array(IEN)



# geracao de um vetor contendo os pontos no contorno e no miolo do dominio

lista_elemen = []
for j in range(len(IEN)):
    if elemento == 'quad':
        for i in range(4):
            lista_elemen.append(IEN[j][i])
    elif elemento == 'triangle':
        for i in range(3):
            lista_elemen.append(IEN[j][i])
        
cnt = Counter(lista_elemen)
maxcnt = max(cnt.values())
inner=[]

for elem in lista_elemen:
    if lista_elemen.count(elem) == maxcnt:
        if elem not in inner:
            inner.append(elem)
            
cc = [x for x in lista_elemen if x not in inner]


# plota malha

xy = np.stack((X, Y), axis=-1)
verts = xy[IEN]
ax=plt.gca()
pc = matplotlib.collections.PolyCollection(verts,edgecolors=('black',),
                                                 facecolors='pink',
                                                 linewidths=(0.7,))
ax.add_collection(pc)
plt.plot(X,Y,'ko')
plt.plot(X[cc],Y[cc],'go')
for i in range(0,npoints):
    plt.text(X[i]+0.02,Y[i]+0.03,str(i),color='b')


plt.gca().set_aspect('equal')
plt.show()



# vizinhos para pontos e elementos

def vizinhoPonto(W, IEN):
    viz = np.array([])
    for elem in IEN:
        if W in elem:
            for I in elem:
                if I!=W:
                    viz = np.append(viz, I)
    return (np.unique(viz))

def vizinhoElem(W, IEN):
    viz = np.array([])
    i=0
    for elem in IEN:        
        if W in elem:
            viz = np.append(viz, i)
        i+=1
    return (np.unique(viz))

listaVizPoint = []
for i in range(npoints):
    listaVizPoint.append(vizinhoPonto(i, IEN).size)

listaVizElem = []
for i in range(npoints):
    listaVizElem.append(vizinhoElem(i, IEN).size)


ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.hist(listaVizPoint, color = 'blue', edgecolor = 'black', bins = 8, label='# pontos vizinhos', range = (0,8))
plt.legend()
plt.show()

plt.hist(listaVizElem, color = 'red', edgecolor = 'black', bins = 8, label='# elementos vizinhos', range = (0,8))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()

def antihorario(IEN):
    for elem in IEN:
        print(elem[0])
        
#teste GIT