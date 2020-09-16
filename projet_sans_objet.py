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
    donne le deuxième plan de la tranche
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
    row est la ligne du dataframe correspondant au Residue
    '''
    x0,y0,z0=tuple(centre_sphere)
    d=np.sqrt((row['x']-x0)**2+(row['y']-y0)**2+(row['z']-z0)**2)
    return d

print(data)

data=data.loc[data.hydrophobe==True]

data['DistanceAuCentre']=data.apply(DistanceAuCentre,axis=1)

DistanceMax=np.abs(data['DistanceAuCentre'].max())

def ListeTranches(Point):
    
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
    
    while DistancePlanBauCentre<DistanceMax:
        
        E=DeterminerPlanSuivant(centre_sphere,E1,EPAISSEUR)
        dTranche=[E1,E]

        E2=PremierPlanProchaineTranche(centre_sphere,E1,Decalage=1)
        E1=E2.copy()

        d1=E1[3]
        Tranches.append(dTranche)
        
        DistancePlanBauCentre=np.abs(a*x0+b*y0+c*z0+d1)/np.sqrt(a**2+b**2+c**2)

    return Tranches

def ListeResiduesDansTranche(E0,E1):
     
    ListeResidues=[]
    
    for index,row in data.iterrows():
        
        Coord_Residue=[row['x'],row['y'],row['z']]
        
        if EstEntre2Plans(Coord_Residue,E0,E1,EPAISSEUR)==True:
            
            ListeResidues.append(row['position'])

    return ListeResidues

#Resultats    
grosse_liste=[]
for index,row in points_sphere_coord.iterrows():
    Point=[row['x'],row['y'],row['z']]
    Tranches=ListeTranches(Point)
    
    Residues=ListeResiduesDansTranche(Tranches[0][0],Tranches[0][1])
    NombreMaxResiduesHydrophobes=[len(Residues)]
    TrancheMax=[Tranches[0]]
    ResiduesMax=[Residues]
    
    for T in Tranches[1:] :

        
        Residues=ListeResiduesDansTranche(T[0],T[1])
        
        if len(Residues)>NombreMaxResiduesHydrophobes[0]:
            NombreMaxResiduesHydrophobes=[len(Residues)]
            TrancheMax=[T]
            ResiduesMax=[Residues]
            
        else :
            if len(Residues)==NombreMaxResiduesHydrophobes[0]:
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

listenontriee=[]
for i in range(len(liste_membrane_pot)):
	for j in range(len(liste_membrane_pot[i])):
		listenontriee.append(liste_membrane_pot[i][j])

#print(listenontriee)
print(sorted(listenontriee))


def couleur(Arg):
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
ax.scatter(data_sauvegarde["x"],data_sauvegarde["y"],data_sauvegarde["z"],c=data_sauvegarde["couleur"])
plt.show()

data_membrane.drop("DistanceAuCentre",axis='columns').to_csv("membrane_resultats_{0}.csv".format(fichier_pdb[-8:-4]),sep=',',index=False)
