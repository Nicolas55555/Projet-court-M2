import os
import argparse
import numpy as np
import pandas as pd
import statistics as stat

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import recup_info_pdb as rip
import sphere

parser = argparse.ArgumentParser()

parser.add_argument('-f', type=str, required=True,
                    help='Chemin d acces au fichier pdb')
parser.add_argument('-s', type=float, default=0.3,
                    help='Seuil d accesibilité au solvant (entre 0 et 1)')
parser.add_argument('-p', type=int, default=50,
                    help='Nombre de points de la sphère')
parser.add_argument('-e', type=float, default=15,
                    help='Largeur de l espace')

args = parser.parse_args()

if os.path.exists(args.f) == False:
    print("Le fichier pdb n'a pas été trouvé")
    exit()

# Paramètres

fichier_pdb = args.f  # ../data/1py6.pdb à donner
SEUIL_ACC = args.s  # 0.3
NB_POINTS_SPHERE = args.p  # 50
EPAISSEUR = args.e  # 15


# Constante(s)

HYDROPHOBE = ["FHE", "GLY", "ILE", "LEU," "MET", "VAL",
              "TRP", "TYR"]  # d'après les données de l'article


# on litle fichier pdb et on récupère les coordonnées des carbones alpha des résidues
data = rip.recup_coord_carbones_alpha(fichier_pdb)
data_sauvegarde = data.copy()
#print(data)

# on set l'hydrophobicité des résidues
data["hydrophobe"] = data.type_aa.isin(HYDROPHOBE)
# print(data["type_aa"])

# on set l'accessiblitéau solvant des résidues
# si pas round dans fonctio,[round(x,3) for x in acces_solvant]
data["acces_solvant"] = rip.recup_acc_solvant(fichier_pdb)

# on garde que les résidues avec plus d'un certain seuil d'acessibilité au solvant pour la sphère
data = data.loc[data['acces_solvant'] > SEUIL_ACC]


######
######


# On récupère le centre de la molécule
centre_sphere = [stat.mean(data.x), stat.mean(data.y), stat.mean(data.z)]

# On crée une sphère
points_sphere_coord = pd.DataFrame(data=sphere.sphere_fibo(
    NB_POINTS_SPHERE), columns=["x", "y", "z"])

# On décalle le centre de la sphère sur le centre de la molécule
points_sphere_coord += centre_sphere

# print(points_sphere_coord)


######
######


def EquationPlan(CentreSphere, PointeVecteur):
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

    vecteur_directeur = [PointeVecteur[i]-CentreSphere[i] for i in range(3)]

    # coordonnées centre sphère
    x0, y0, z0 = tuple(CentreSphere)

    # coordonnées extrémité du vecteur
    x1, y1, z1 = tuple(PointeVecteur)

    # paramètres de l'équation du plan
    a, b, c = tuple(vecteur_directeur)
    d = -(a*x1+b*y1+c*z1)
    Parametres = vecteur_directeur+[d]

    return Parametres


def DeterminerPlanSuivant(CentreSphere, Parametres, EPAISSEUR):
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
    a, b, c, d = tuple(Parametres)

    x0, y0, z0 = tuple(CentreSphere)

    # distance du plan de départ au centre de la sphère
    DistancePlanCentre = np.abs(a*x0+b*y0+c*z0+d)/np.sqrt(a**2+b**2+c**2)

    # k : facteur reliant les vecteurs normaux aux plans
    k = (DistancePlanCentre+EPAISSEUR)/np.sqrt(a**2+b**2+c**2)

    x1 = k*a+x0
    y1 = k*b+y0
    z1 = k*c+z0

    d1 = -(a*x1+b*y1+c*z1)

    Parametres = [a, b, c, d1]

    return Parametres


def PremierPlanProchaineTranche(CentreSphere, Parametres, Decalage=1):
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
    a, b, c, d = tuple(Parametres)

    x0, y0, z0 = tuple(CentreSphere)

    # distance du plan de départ au centre de la sphère
    DistancePlanCentre = np.abs(a*x0+b*y0+c*z0+d)/np.sqrt(a**2+b**2+c**2)

    # k : facteur reliant les vecteurs normaux aux plans
    k = (DistancePlanCentre+Decalage)/np.sqrt(a**2+b**2+c**2)

    x1 = k*a+x0
    y1 = k*b+y0
    z1 = k*c+z0

    d1 = -(a*x1+b*y1+c*z1)

    Parametres = [a, b, c, d1]

    return Parametres


def EstEntre2Plans(Coord_Residue, ParametresA, ParametresB, EPAISSEUR):
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
    if ParametresA[0:2] != ParametresB[0:2]:
        return ('Error : Les deux plans ne sont pas parallèles')

    xR, yR, zR = tuple(Coord_Residue)

    a, b, c, d = tuple(ParametresA)
    d1 = ParametresB[3]

    DistancePlanA = np.abs(a*xR+b*yR+c*zR+d)/np.sqrt(a**2+b**2+c**2)
    DistancePlanB = np.abs(a*xR+b*yR+c*zR+d1)/np.sqrt(a**2+b**2+c**2)

    if DistancePlanA < EPAISSEUR and DistancePlanB < EPAISSEUR:
        return True
    else:
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

    x0, y0, z0 = tuple(centre_sphere)
    d = np.sqrt((row['x']-x0)**2+(row['y']-y0)**2+(row['z']-z0)**2)
    return d


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
    x0, y0, z0 = tuple(centre_sphere)

    # premier plan correspondant au vecteur:
    E0 = EquationPlan(centre_sphere, Point)
    E1 = DeterminerPlanSuivant(centre_sphere, E0, EPAISSEUR)

    dTranche0 = [E0, E1]

    Tranches = [dTranche0]

    a, b, c, d = tuple(E0)

    E2 = PremierPlanProchaineTranche(centre_sphere, E0, Decalage=1)
    E1 = E2.copy()

    d1 = E1[3]

    DistancePlanBauCentre = np.abs(a*x0+b*y0+c*z0+d1)/np.sqrt(a**2+b**2+c**2)

    # DistanceMax est la distance au centre du Residue le plus éloigné.
    while DistancePlanBauCentre < DistanceMax:

        E = DeterminerPlanSuivant(centre_sphere, E1, EPAISSEUR)
        dTranche = [E1, E]

        E2 = PremierPlanProchaineTranche(centre_sphere, E1, Decalage=1)
        E1 = E2.copy()

        d1 = E1[3]
        Tranches.append(dTranche)

        DistancePlanBauCentre = np.abs(
            a*x0+b*y0+c*z0+d1)/np.sqrt(a**2+b**2+c**2)

    return Tranches


def ListeResiduesDansTranche(E0, E1):
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

    ListeResidues = []

    for index, row in data.iterrows():

        Coord_Residue = [row['x'], row['y'], row['z']]

        if EstEntre2Plans(Coord_Residue, E0, E1, EPAISSEUR) == True:

            ListeResidues.append(row['position'])

    return ListeResidues


# Les lignes qui suivent définissent la distance entre le centre de la sphère et le Residue le plus éloigné.
# On définit aussi les tranches les plus éloignées du centre de la sphère.

data = data.loc[data.hydrophobe == True]

data['DistanceAuCentre'] = data.apply(DistanceAuCentre, axis=1)

DistanceMax = np.abs(data['DistanceAuCentre'].max())


# Resultats
residues_tranche_max = []
for index, row in points_sphere_coord.iterrows():

    # Cette boucle parcourt les points de la sphère, et donc les vecteurs qu'ils définissent,
    # pour déterminer les Residues hydrophobes compris dans chaque tranche normale à ces vecteurs.
    # Le résultat est une liste contenant tous les Residues trouvés dans les tranches en contenant le plus grand nombre pour chaque vecteur.

    Point = [row['x'], row['y'], row['z']]
    Tranches = ListeTranches(Point)

    Residues = ListeResiduesDansTranche(Tranches[0][0], Tranches[0][1])
    # Nombre maximal de Residues hydrophobes dans une tranche, selon un vecteur donné. Sera mis à jour dans la boucle suivante.
    NombreMaxResiduesHydrophobes = [len(Residues)]
    # Paramètres de la tranche contenant le plus grand nombre de Residues hydrophobes. Sera mis à jour dans la boucle suivante.
    TrancheMax = [Tranches[0]]
    # Liste des Residues contenus dans la tranche TrancheMax. Sera mis à jour dans la boucle suivante.
    ResiduesMax = [Residues]

    for T in Tranches[1:]:

        # Cette boucle parcourt les tranches le long d'un vecteur donné pour définir celles qui contiennent le plus grand nombre de Residues hydrophobes.

        Residues = ListeResiduesDansTranche(T[0], T[1])

        if len(Residues) > NombreMaxResiduesHydrophobes[0]:
            NombreMaxResiduesHydrophobes = [len(Residues)]
            TrancheMax = [T]
            ResiduesMax = [Residues]

        else:
            if len(Residues) == NombreMaxResiduesHydrophobes[0]:
                # Cette condition permet de tenir compte des cas où plusieurs tranches ont le nombre maximal de Residues hydrophobes le long d'un vecteur donné.
                TrancheMax.append(T)
                ResiduesMax.append(Residues)
    for i in range(len(ResiduesMax)):
        residues_tranche_max.append(ResiduesMax[i])

    # print('NMax',NombreMaxResiduesHydrophobes,'nombreTranchesMemeNombre',len(TrancheMax))
    # print(ResiduesMax)

longueur = []
for i in residues_tranche_max:
    longueur.append(len(i))

liste_membrane_pot = []
long_max = max(longueur)

for i in residues_tranche_max:
    if len(i) == (long_max):
        liste_membrane_pot.append(i)

# Liste de tous les Residues trouvés précédemment, non triés selon le vecteur normal à leur tranche.
listenontriee = []
for i in range(len(liste_membrane_pot)):
    for j in range(len(liste_membrane_pot[i])):
        listenontriee.append(liste_membrane_pot[i][j])


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
    else:
        return 'b'


data_membrane = data.loc[data['position'].isin(residues_tranche_max[0])]
data_sauvegarde["membrane"] = data_sauvegarde.position.isin(
    data_membrane.position)
data_sauvegarde["couleur"] = data_sauvegarde.membrane.apply(couleur)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data["x"],data["y"],data["z"])

# Représentation de la protéine, les Residues hydrophobes sélectionnés précédemment sont en rouge.
ax.scatter(data_sauvegarde["x"], data_sauvegarde["y"],
           data_sauvegarde["z"], c=data_sauvegarde["couleur"])
#plt.show()
fig.savefig('../results/{0}_{1}_{2}_{3}.png'.format(fichier_pdb[-8:-4],SEUIL_ACC,NB_POINTS_SPHERE,EPAISSEUR))


# Enregistrement des résultats au format csv
data_membrane.drop("DistanceAuCentre", axis='columns').to_csv(
    "../results/membrane_{0}_{1}_{2}_{3}.csv".format(fichier_pdb[-8:-4],SEUIL_ACC,NB_POINTS_SPHERE,EPAISSEUR), sep=',', index=False)
