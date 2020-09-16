import os
import argparse
import numpy as np
import pandas as pd
import statistics as stat 

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()

parser.add_argument('-f', type=str, required=True, 
					help='Chemin d acces au fichier pdb')
parser.add_argument('-s', type=float, default=0.3, 
					help='Seuil d accesibilité au solvant (entre 0 et 1)')
parser.add_argument('-p', type=int, default=256,
	 				help='Nombre de points de la sphère')
parser.add_argument('-e',type=float, default=15, 
					help='Largeur de l espace')

args = parser.parse_args()

if os.path.exists(args.f) == False:
	print("Le fichier pdb n'a pas été trouvé")
	exit()

##Paramètres

fichier_pdb = args.f #../data/1py6.pdb à donner
SEUIL_ACC = args.s #0.3
NB_POINTS_SPHERE = args.p #256
EPAISSEUR = args.e #15


##Constante(s)

HYDROPHOBE=["FHE", "GLY", "ILE", "LEU," "MET", "VAL", "TRP","TYR"] # d'après les données de l'article 

def recup_coord_carbones_alpha(fichier_pdb):
	"""
		Recupération des coordonnées des carbones alpha 
		à partir du fichier pdb donné en entrée
		et renvoie d'un dataframe
	"""
	data_pdb = []
	file = open(fichier_pdb, "r")
	
	for ligne in file:
		if ligne.startswith("ATOM"):
			if (str.strip(ligne[12:15]) == "CA" and 
			(str.strip(ligne[16:17]) == "" or str.strip(ligne[16:17]) == "A")):
				data_pdb.append([(ligne[21:22],
								int(str.strip(ligne[22:26]))),
								ligne[17:20],
				                float(str.strip(ligne[30:38])),
				                float(str.strip(ligne[38:46])),
				                float(str.strip(ligne[46:54]))])

	file.close()
 
	data = pd.DataFrame(data = data_pdb, columns = ["position","type_aa", "x", "y", "z"])
 
	return data

def recup_acc_solvant(fichier_pdb):
	"""
	entrée le fichier pdb
	sortie une liste de l'accesibilité relative (entre 0 et 1) des résidues
	utilise Biopython et le programme dssp

	"""
	acces_solvant = []
	p = PDBParser()
	id_structure = fichier_pdb.split(".")[0]

	structure = p.get_structure(id_structure, fichier_pdb)
	model = structure[0]
	dssp = DSSP(model, fichier_pdb, dssp = 'mkdssp')
 
	for i in range(len(list(dssp))):
		acces_solvant.append(round((list(dssp)[i][3]),3)) #ajout de chaque élément arrondi au millième

	return acces_solvant

#on litle fichier pdb et on récupère les coordonnées des carbones alpha des résidues
data = recup_coord_carbones_alpha(fichier_pdb) 
data_sauvegarde=data.copy()
print(data)

#on set l'hydrophobicité des résidues
data["hydrophobe"] = data.type_aa.isin(HYDROPHOBE)
#print(data["type_aa"])

#on set l'accessiblitéau solvant des résidues
data["acces_solvant"] = recup_acc_solvant(fichier_pdb) ## si pas round dans fonctio,[round(x,3) for x in acces_solvant]

#on garde que les résidues avec plus d'un certain seuil d'acessibilité au solvant pour la sphère
data = data.loc[data['acces_solvant'] > SEUIL_ACC]



######
######

def recup_centre_sphere(data_frame):
	return([stat.mean(data.x),stat.mean(data.y),stat.mean(data.z)])


def sphere_fibo(nb_points):
	"""
	Entrée le nombre de points de la spère
	créé des points uniformement repartis sur la sphère de centre (0,0,0)
	Renvoie un numpy.array des points

	"""
	golden_angle = np.pi * (3 - np.sqrt(5))
	theta = golden_angle * np.arange(nb_points)
	z = np.linspace(1 - 1.0 / nb_points, 1.0 / nb_points - 1, nb_points)
	radius = np.sqrt(1 - z * z)
 
	points = np.zeros((nb_points, 3))
	points[:,0] = radius * np.cos(theta)
	points[:,1] = radius * np.sin(theta)
	points[:,2] = z
  
	return points
	
#On récupère le centre de la molécule
centre_sphere=recup_centre_sphere(data)


#On crée une sphère 
points_sphere_coord = pd.DataFrame(data = sphere_fibo(NB_POINTS_SPHERE), columns = ["x", "y", "z"])

#On décalle le centre de la sphère sur le centre de la molécule
points_sphere_coord += centre_sphere 

##print(points_sphere_coord)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_sphere_coord["x"],points_sphere_coord["y"],points_sphere_coord["z"])
#plt.show()

################################


def EquationPlan(CentreSphere,PointeVecteur):
    
	'''
	Retourne les paramètres a,b,c,d définissant l'équation de plan de type a*x+b*y+c*z+d=0, avec x,y,z des nombres réels.
	
	ARGUMENTS
	----------
	CentreSphere : coordonnées du centre de la sphère définie plus haut, au format [x0,y0,z0].
	PointeVecteur : coordonnées de l'extrémité du vecteur normal au plan, au format [x1,y1,z1].
	
	SORTIES
	----------
	Parametres=[a,b,c,d]
	
	'''
	
    vecteur_directeur=[PointeVecteur[i]-CentreSphere[i] for i in range(3)]
    
    #coordonnées centre sphère
    x0,y0,z0=tuple(CentreSphere)

    #coordonnées extrémité du vecteur
    x1,y1,z1=tuple(PointeVecteur)
    
    #paramètres de l'équation du plan
    a,b,c=tuple(vecteur_directeur)
    d=-(a*x1+b*y1+c*z1)
    Parametres=vecteur_directeur+[d]
    
    return Parametres

def DeterminerPlanSuivant(CentreSphere,Parametres,EPAISSEUR):
   	'''
	Dans le but de définir une "tranche" dans l'espace composée de deux plans, cette fonction définit les paramètres du second plan en fonction du premier.
	
	ARGUMENTS
	----------
	CentreSphere : coordonnées du centre de la sphère définie plus haut, au format [x0,y0,z0].
	Parametres : parametres au format [a,b,c,d] de l'équation définissant le premier plan.
	EPAISSEUR : distance entre les deux plans parallèles.
	
	SORTIES
	----------
	Parametres : au format [a,b,c,d1], définissant l'équation du second plan.
	
	'''
    a,b,c,d=tuple(Parametres)

    x0,y0,z0=tuple(CentreSphere)
    
    #distance du plan de départ au centre de la sphère
    DistancePlanCentre=np.abs(a*x0+b*y0+c*z0+d)/np.sqrt(a**2+b**2+c**2)

    k=(DistancePlanCentre+EPAISSEUR)/np.sqrt(a**2+b**2+c**2) # k : facteur reliant les vecteurs normaux aux plans

    x1=k*a+x0
    y1=k*b+y0
    z1=k*c+z0

    d1=-(a*x1+b*y1+c*z1)
    
    Parametres=[a,b,c,d1]

    return Parametres


def PremierPlanProchaineTranche(CentreSphere,Parametres,Decalage=1):
    
	'''
	A partir du premier plan d'une tranche, définit les paramètres d'un plan parallèle, à partir duquel sera définie la prochaine tranche.
	
	ARGUMENTS
	----------
	CentreSphere : coordonnées du centre de la sphère définie plus haut, au format [x0,y0,z0].
	Parametres : parametres au format [a,b,c,d] de l'équation définissant le premier plan de la tranche précédente.
	Decalage : distance entre les deux plans parallèles.
	
	SORTIES
	----------
	Parametres : au format [a,b,c,d1], définissant l'équation du premier plan de la tranche suivante.
	
	'''
    a,b,c,d=tuple(Parametres)

    x0,y0,z0=tuple(CentreSphere)
    
    #distance du plan de départ au centre de la sphère
    DistancePlanCentre=np.abs(a*x0+b*y0+c*z0+d)/np.sqrt(a**2+b**2+c**2)

    k=(DistancePlanCentre+Decalage)/np.sqrt(a**2+b**2+c**2) # k : facteur reliant les vecteurs normaux aux plans

    x1=k*a+x0
    y1=k*b+y0
    z1=k*c+z0

    d1=-(a*x1+b*y1+c*z1)
    
    Parametres=[a,b,c,d1]
    
    return Parametres


def EstEntre2Plans(Coord_Residue,ParametresA,ParametresB,EPAISSEUR):
	'''
	Détermine si un point donné se situe entre deux plans parallèles distincts d'une épaisseur donnée.
	
	ARGUMENTS
	----------
	Coord_Residue : coordonnées du point étudié, au format [xR,yR,zR].
	ParametresA : parametres au format [a,b,c,d] de l'équation définissant le premier plan de la tranche.
	ParametresB : parametres au format [a,b,c,d1] de l'équation définissant le second plan de la tranche.
	EPAISSEUR : float définissant l'écart voulu entre les plans de chaque tranche.
	
	SORTIES
	----------
	Booléen : True si le point se situe entre les plans parallèles ; False sinon.
	
	'''
    if ParametresA[0:2]!=ParametresB[0:2] :
        return ('Error : Les deux plans ne sont pas parallèles')
    
    xR,yR,zR=tuple(Coord_Residue)

    a,b,c,d=tuple(ParametresA)
    d1=ParametresB[3]
    
    DistancePlanA=np.abs(a*xR+b*yR+c*zR+d)/np.sqrt(a**2+b**2+c**2)
    DistancePlanB=np.abs(a*xR+b*yR+c*zR+d1)/np.sqrt(a**2+b**2+c**2)
  
    if DistancePlanA<EPAISSEUR and DistancePlanB<EPAISSEUR:
        return True
    else :
        return False



def DistanceAuCentre(row): 
    '''
	Donne la distance au centre de la sphère d'un point donné.
	
	ARGUMENTS
	----------
	row : ligne d'un Pandas.DataFrame contenant les colonnes 'x', 'y' et 'z' correspondant aux coordonnées du point étudié.
	
	SORTIES
	----------
	d : distance entre le point étudié et le centre de la sphère.
    '''
	
    x0,y0,z0=tuple(centre_sphere)
    d=np.sqrt((row['x']-x0)**2+(row['y']-y0)**2+(row['z']-z0)**2)
    return d

print(data)


# Les lignes qui suivent définissent la distance entre le centre de la sphère et le Residue le plus éloigné.
# Cela définit plus loin les tranches les plus éloignées du centre de la sphère.

data=data.loc[data.hydrophobe==True]

data['DistanceAuCentre']=data.apply(DistanceAuCentre,axis=1)

DistanceMax=np.abs(data['DistanceAuCentre'].max())

def ListeTranches(Point):
    '''
	Etablit une liste des tranches de l'espace (entre deux plans), parallèles entre elles et normales à un vecteur.
	
	ARGUMENTS
	----------
	Point : coordonnées au format [x0,y0,z0] d'un point de la sphère correspondant à l'extrémité du vecteur qui définit les plans.
	
	SORTIES
	----------
	Tranches : liste de coordonnées des deux plans formant chaque tranche, 
	au format [E1,E], où E1 et E sont les paramètres de chaque plan, au format [a,b,c,d].
    '''
    x0,y0,z0=tuple(centre_sphere)
	
    #premier plan correspondant au vecteur:
    E0=EquationPlan(centre_sphere,Point)
    E1=DeterminerPlanSuivant(centre_sphere,E0, EPAISSEUR)
	
    dTranche0=[E0,E1]
    
    Tranches=[dTranche0]
    
    a,b,c,d=tuple(E0)

    E2=PremierPlanProchaineTranche(centre_sphere,E0,Decalage=1)
    E1=E2.copy()

    d1=E1[3]
    
    DistancePlanBauCentre=np.abs(a*x0+b*y0+c*z0+d1)/np.sqrt(a**2+b**2+c**2)
    
    while DistancePlanBauCentre<DistanceMax: #DistanceMax est la distance au centre du Residue le plus éloigné.
        
        E=DeterminerPlanSuivant(centre_sphere,E1,EPAISSEUR)
        dTranche=[E1,E]

        E2=PremierPlanProchaineTranche(centre_sphere,E1,Decalage=1)
        E1=E2.copy()

        d1=E1[3]
        Tranches.append(dTranche)
        
        DistancePlanBauCentre=np.abs(a*x0+b*y0+c*z0+d1)/np.sqrt(a**2+b**2+c**2)

    return Tranches

def ListeResiduesDansTranche(E0,E1):
    '''
	A partir de deux plans définissant une tranche de l'espace, détermine les Residues compris entre ces deux plans.
	
	ARGUMENTS
	----------
	E0 : paramètres au format [a,b,c,d] du premier plan définissant la tranche.
	E1 : paramètres au format [a,b,c,d] du second plan définissant la tranche.
	
	SORTIES
	----------
	ListeResidues : liste contenant les positions dans la chaîne des Residues compris entre les deux plans.
	
	'''
	 
    ListeResidues=[]
    
    for index,row in data.iterrows():
        
        Coord_Residue=[row['x'],row['y'],row['z']]
        
        if EstEntre2Plans(Coord_Residue,E0,E1,EPAISSEUR)==True:
            
            ListeResidues.append(row['position'])

    return ListeResidues

#Resultats    
grosse_liste=[]
for index,row in points_sphere_coord.iterrows():

	# Cette boucle parcourt les points de la sphère, et donc les vecteurs qu'ils définissent, 
	# pour déterminer les Residues hydrophobes compris dans chaque tranche normale à ces vecteurs.
	# Le résultat est une liste contenant tous les Residues trouvés dans les tranches en contenant le plus grand nombre pour chaque vecteur.
	
    Point=[row['x'],row['y'],row['z']]
    Tranches=ListeTranches(Point)
    
    Residues=ListeResiduesDansTranche(Tranches[0][0],Tranches[0][1])
    NombreMaxResiduesHydrophobes=[len(Residues)] #Nombre maximal de Residues hydrophobes dans une tranche, selon un vecteur donné. Sera mis à jour dans la boucle suivante.
    TrancheMax=[Tranches[0]] #Paramètres de la tranche contenant le plus grand nombre de Residues hydrophobes. Sera mis à jour dans la boucle suivante.
    ResiduesMax=[Residues] #Liste des Residues contenus dans la tranche TrancheMax. Sera mis à jour dans la boucle suivante.
    
    for T in Tranches[1:] :
	
		# Cette boucle parcourt les tranches le long d'un vecteur donné pour définir celles qui contiennent le plus grand nombre de Residues hydrophobes.
        
        Residues=ListeResiduesDansTranche(T[0],T[1])
        
        if len(Residues)>NombreMaxResiduesHydrophobes[0]:
            NombreMaxResiduesHydrophobes=[len(Residues)]
            TrancheMax=[T]
            ResiduesMax=[Residues]
            
        else :
            if len(Residues)==NombreMaxResiduesHydrophobes[0]:
				# Cette condition permet de tenir compte des cas où plusieurs tranches ont le nombre maximal de Residues hydrophobes le long d'un vecteur donné.
                TrancheMax.append(T)
                ResiduesMax.append(Residues)
    for i in range(len(ResiduesMax)):
    	grosse_liste.append(ResiduesMax[i])


    #print('NMax',NombreMaxResiduesHydrophobes,'nombreTranchesMemeNombre',len(TrancheMax))
    #print(ResiduesMax)

longueur=[]
print(len(grosse_liste))
for i in grosse_liste:
	longueur.append(len(i))

liste_membrane_pot=[]
long_max=max(longueur)

for i in grosse_liste:
	if len(i)==(long_max):
		liste_membrane_pot.append(i)

print("salut")
print(liste_membrane_pot)

for i in liste_membrane_pot:
	print(len(i))

listenontriee=[] #Liste de tous les Residues trouvés précédemment, non triés selon le vecteur normal à leur tranche.
for i in range(len(liste_membrane_pot)):
	for j in range(len(liste_membrane_pot[i])):
		listenontriee.append(liste_membrane_pot[i][j])

#print(listenontriee)
print(sorted(listenontriee))


def couleur(Arg):
	'''
	Détermine la couleur d'une donnée en fonction de sa valeur True ou False.
	Cette fonction est utilisée pour donner une couleur rouge aux Residues obtenus dans cette étude, en vue de la réalisation de figures.
	
	ARGUMENTS
	----------
	Arg : Booléen True ou False.
	
	SORTIES
	----------
	'r' : définit la couleur rouge si l'entrée est 'True'.
	'b' : définit la couleur bleue si l'entrée est 'False'.
	
	'''
	if Arg == True:
		return 'r'
	else :
		return 'b'


data_membrane=data.loc[data['position'].isin(grosse_liste[0])]
data_sauvegarde["membrane"]=data_sauvegarde.position.isin(data_membrane.position)
data_sauvegarde["couleur"]=data_sauvegarde.membrane.apply(couleur)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(data["x"],data["y"],data["z"])

# Représentation de la protéine, les Residues hydrophobes sélectionnés précédemment sont en rouge.
ax.scatter(data_sauvegarde["x"],data_sauvegarde["y"],data_sauvegarde["z"],c=data_sauvegarde["couleur"])
plt.show()

# Enregistrement des résultats au format csv
data_membrane.drop("DistanceAuCentre",axis='columns').to_csv("membrane_resultats_{0}.csv".format(fichier_pdb[-8:-4]),sep=',',index=False)
