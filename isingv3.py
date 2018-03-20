# -*- coding: utf-8 -*
""" 
Objectif : obtenir une simulation d'un modèle d'Ising (c'est a dire un ensemble de spin en interaction sur réseau )


Première implémentation :
On part de l'algorithme métropolis de base:
On dispose de deux grilles ( en pratique 3 pour gagner du temps de calcul ), une contenant n^2 valeurs de spin \in {-1,+1} (=> Raffinement possible dans Potss assez facile ),  et l'autre contenant n^2 valeur d'énergie, c'est a dire le terme 
de l'hamiltonien classique du modèle d'Ising pour le spin situé en i,j. 
La troisième grille ne sert qu'a éviter des calculs : il s'agit d'une matrice n*n contenant des listes de couple. Si on note Neighbours_Grid cette matrice, Neighbours_Grid_{i,b} =  Matrice des positions des voisins. La dimension 
de cette matrice est nb_voisins*2. Les conditions aux bords consistent a considérer que le milieu ne dépasse pas de  la matrice testé. Elle n'est calculé qu'une unique fois. L'intérêt est de ne pas passer au travers soit de modulo soit 
	- On part d'un état connu pour Spin_Grid 
	- On sélectionne un point au hasard dans la matrice. 
	- On change la valeur de spin a cette endroit.
	- On recalcule l'énergie associé au spin changé, c'est a dire celle du spin elle même et celle de ses 4 voisins. On les stocke dans une copie de la matrice d'énergie. 
	- Ensuite, on regarde si le changement est intéressant énergétiquement  ou non : on change le spin avec une probabilité associé au poids de Boltzmann  : P(E_\mu \rightarrow E_\lambda) = $\exp{-\dfrac{E_\lambda - E_\mu]{k_b T}}
	- On réitère l'opération un nombre de fois suffisant pour assurer que la l'état final soit décorelllé de l'état initial : c'est une manière de quantifier l'écart à l'équilibre.

Remarque  : on se préocupe de la vitesse d'exécution, la mémoire vive n'est a priori pas le facteur limitant. ( ce qui justifie l'utilisation d'une 3ème grille pour les voisins, plutot que des les recalculer. )


On attend l'existence d'une température critique dans le cadre d'un modèle ferromagnétiuqe: il s'agit de la température telle qu'on ai une transition de phase ferromagnetique vers paramagnétique. Il s'agit de la température telle que l'énergie
d'agitation thermique deviennnent grande devant l'énergie d'interaction. 
Cette approche rencontre deux problèmes  : 
Les temps de convergence deviennent très long au niveau de la température critique.
On cherche a miniser l'énergie : on peut donc assister se retrouver dans un minimum local d'énergie et  si la température est suffisamment faible on peut se retrouver coincé dedans.  

On passe donc a la version de Wolff et al.
L'idée ici est de ne plus changer un spin, mais un cluster.L'algorithme est le même que précédement, a ceci près qu'on forme un cluster et qu'on le change, avec la même probabilité. 
Pour former ce cluster, on se munit d'une 4ème grille de la même taille, initialisé a np.nan 
- On choisit un point i,j dans la grille, qui sera la graine du cluster. State_Grid[i,j] = True

Algorithme récursif de formation de cluster :
- On regarde les voisins les plus proches. On se donne un voisin [k,l] : si ce voisin n'a pas été pas testé (State_Grid = np.nan ), que Spin_Grid [k,l] == Spin_Grid[i,j], alors on  a 
State_Grid [k,l] = True avec proba p et on ajoute ce voisin a la liste des points dont on va devoir tester les voisins  par la suite. Sinon Stage_Grid[k,l] = False, et on ne testera pas ses voisins.

L'intérêt est de changer un nombre de spins plus importants a chaque itération. Cela permet d'améliorer la vitesse de convergence et éventuellement de réussir a sortir de certain minima locaux d'énergie. 


Bibliographie  : 
Onsager, Lars (1944), "Crystal statistics. I. A two-dimensional
model with an order-disorder transition", Phys. Rev. (2) 65
(3–4): 117–149 

https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.62.361  


"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from multiprocessing import Pool
import numba 


kB=1.3806e-23 #J/K
uB=9.274e-24


### On définit toutes les fonctions utilitaires ici 

def Init_Grid(Grid_size):
    """
    Création du réseau et initialisation sur un état de spin aléatoire.
    """
    A = np.random.rand(Grid_size[0]*Grid_size[1])
    A= A.reshape([Grid_size[0],Grid_size[1]])
    A=np.round(A)
    A[A==0] = -1
    return A 

def Hamiltonien(A,J,h): 
    """
    Calcul du Halmitonien pour un réseau donné.Ne sert a priori qu'une fois par intération
    """
    Kernel = np.array([[0,1,0],[1,0,1],[0,1,0]])
    M=-J*A*ss.convolve2d(A,Kernel,mode='same')-uB*h*A
    return M



    

  

def return_neighbours(R,C,i,j):
    if i !=R-1 and i != 0 and j != 0 and j !=C-1:
        List_Neighbours=np.array([[i,j+1],[i,j-1],[i+1,j],[i-1,j]])
    elif i == R-1 and j != 0 and j !=C-1: 
        List_Neighbours=np.array([[i,j+1],[i,j-1],[i-1,j]])
    elif i==0 and j !=0 and j!= C-1 :
        List_Neighbours=np.array([[i,j+1],[i,j-1],[i+1,j]] )
    elif j==0 and i !=0 and i!= R-1 :
     	 List_Neighbours=np.array([[i,j+1],[i+1,j],[i-1,j]] )
    elif j==C-1 and i !=0 and i!= R-1 :
    	 List_Neighbours=np.array([[i,j-1],[i+1,j],[i-1,j]] )
    elif i==0 and j == 0 : 
        List_Neighbours=np.array([[i,j+1],[i+1,j]] )
    elif i==0 and j ==C-1 : 
        List_Neighbours=np.array([[i,j-1],[i+1,j]] )
    elif i==R-1 and j == R-1 : 
    	 List_Neighbours=np.array([[i,j-1],[i-1,j]] )
    elif i == R-1 and j == 0 : 
    	 List_Neighbours=np.array([[i,j+1],[i-1,j]] )
    return List_Neighbours



def create_Neighbours_Grid(Spin_Grid):
	"""
		Crée la matrice contenant la liste des voisins.!!! N'était pas possible de faire des matrices de matrices et monter en dimension n'est pas possible, on ne stocke pas 
        pas dans une matrice numpy mais dans une liste python classique. Les voisins de l'élément i,j sont a la position i*R+j  dans la liste par construction.. 
	"""
	R,C = Spin_Grid.shape
	Neighbours_Grid= []
	for i in range(R):
		for j in range(C): 
			Neighbours_Grid.append(return_neighbours(R,C,i,j))
	return Neighbours_Grid


def update_energy(Spin_Grid,Neighbours_Grid,i,j,J,h,T):
    """
    Renvoie l'énergie associé au spin i,j, interagissant avec ses voisins. 
    """
    E= - h* uB*Spin_Grid[i,j]
    R,C = Spin_Grid.shape
    List_Neighbours = Neighbours_Grid[R*i+j]
    for k in List_Neighbours:
        E += - J*Spin_Grid[i,j]*Spin_Grid[k[0],k[1]]
    return E 







def update_Grid(Spin_Grid,Neighbours_Grid,State_Grid,Energy_Grid,J,h,T):
    """ 
         Responsable de mettre a jour la grille d'énergie après la modification de i,j. Elle en retourne une copie. On pourrait technique se passer de la copie et ne 
    """
    Updated_Energy_Grid = np.copy(Energy_Grid) 
    R,C = Spin_Grid.shape
    to_update = [Neighbours_Grid[R*k[0]+k[1]] for k in np.argwhere(State_Grid)]
    for k in to_update:
        for j in k :
            Updated_Energy_Grid[j[0],j[1]] = update_energy(Spin_Grid,Neighbours_Grid,j[0],j[1],J,h,T)
    return Updated_Energy_Grid





### Routine de création de cluster. Cette version de l'algorithme n'impose pas la copie 

def Create_Cluster(Spin_Grid,Neighbours_Grid,T,treshold):    
    State_Grid = np.full(Spin_Grid.shape, np.nan)
    R,C = Spin_Grid.shape
    i,j = np.random.randint(0,R),np.random.randint(0,R)
    State_Grid[i,j] = True
    add_neighbours_to_clusters(State_Grid,Spin_Grid,Neighbours_Grid,i,j,T,treshold)
    State_Grid[np.isnan(State_Grid)] =False
    return State_Grid

def add_neighbours_to_clusters(State_Grid,Spin_Grid,Neighbours_Grid,i,j,T,treshold):
    R,C = Spin_Grid.shape
    List_Neighbours = Neighbours_Grid[R*i+j]
    Test_value = np.random.random([len(List_Neighbours)])
    to_test =[]
    for k,l in zip(List_Neighbours,Test_value):
        if l < treshold and Spin_Grid[i,j] == Spin_Grid[k[0],k[1]] and np.isnan(State_Grid[k[0],k[1]])  and State_Grid[i,j] == True: 
            to_test.append(k)
            State_Grid[k[0],k[1]] = True 
        elif np.isnan(State_Grid[k[0],k[1]]) :
            State_Grid[k[0],k[1]] = False
    for k in to_test:
    	add_neighbours_to_clusters(State_Grid,Spin_Grid,Neighbours_Grid,k[0],k[1],T,treshold)
    return(None) 




def Iteration_Ising(Spin_Grid,Energy_Grid,Neighbours_Grid,J,H,T,treshold ):
    """
    Réalise une itération dans le modèle d'Ising
    """
    State_Grid=(Create_Cluster(Spin_Grid,Neighbours_Grid,T,treshold))
    for k in np.argwhere(State_Grid): 
        Spin_Grid[k[0],k[1]]= -Spin_Grid[k[0],k[1]]
    Updated_Energy_Grid = update_Grid(Spin_Grid,Neighbours_Grid,State_Grid,Energy_Grid,J,H,T)
    Delta = np.sum(Updated_Energy_Grid-Energy_Grid)
    beta = 1/(kB*T)
    x=np.random.random()
    if x<np.exp(-Delta*beta):
        return Spin_Grid,Updated_Energy_Grid
    else : 
        for k in np.argwhere(State_Grid): 
            Spin_Grid[k[0],k[1]]= -Spin_Grid[k[0],k[1]]
        return (Spin_Grid,Energy_Grid)






### A finir de réecrire 

numba.jit(parallel = True)   
def Ising(Grid_size,Nb_iter,Jcoupling,Hmagnetic,Temp,treshold):
    Spin_Grid = Init_Grid(Grid_size)
    Energie_Grid = Hamiltonien(Spin_Grid,Jcoupling,Hmagnetic)
    original = np.copy(Spin_Grid)
    Neighbours_Grid=create_Neighbours_Grid(Spin_Grid)
    Magnetisation=np.zeros([Nb_iter,1])
    Energie=np.zeros([Nb_iter,1])
    Energie[0,0]= np.sum(Energie_Grid)
    Magnetisation[0,0]=np.sum(Spin_Grid)/Spin_Grid.size
    for k in range(1,Nb_iter):
        Spin_Grid,Energie_Grid = Iteration_Ising(Spin_Grid,Energie_Grid,Neighbours_Grid, Jcoupling,Hmagnetic,Temp,treshold)
        Energie[k,0] = np.sum(Energie_Grid)
        Magnetisation[k,0]=np.sum(Spin_Grid)/Spin_Grid.size
    np.savez("T=" + str(Temp),original,Spin_Grid,Energie,Magnetisation)
    print(Hmagnetic,Temp,"done")
    return original,Spin_Grid,Energie,Magnetisation
    
def Ising_multiprocess(Entree):
    Original,Final,Energie,Magnetisation =Ising([10,10],10000000,6e-22,0,Entree)
    return Magnetisation


if __name__ == '__main__':
    Init,Final,E,M =Ising([50,50],1000000,6e-22,0,200,0.34)
    print(Final)
    plt.plot(M)
    plt.show()
    plt.imshow(Final)
    plt.show()