
import pandas as pd
import ssbio

def recup_coord_carbones_alpha(fichier_pdb):
	"""
		Recupération des coordonnées des carbones alpha à partir du fichier pdb 
		sous forme d'une grand liste, conversion sous forme de dataframe et renvoie du dataframe
	"""
	data_pdb=[]
	file=open(fichier_pdb,"r")
	
	for ligne in file:
		if ligne.startswith("ATOM"):
			if str.strip(ligne[12:15])=="CA" and (str.strip(ligne[16:17])=="" or str.strip(ligne[16:17])=="A"):
				data_pdb.append([ligne[17:20],float(str.strip(ligne[30:37])),float(str.strip(ligne[38:46])),float(str.strip(ligne[46:53]))])
	file.close()

	data=pd.DataFrame(data=data_pdb,columns=["type_aa","x","y","z"])

	return (data)

data=recup_coord_carbones_alpha("data/pdb/5jsi.pdb") 

print(data)

#d'après l'article aa hydrophobes
hydrophobe=["FHE", "GLY", "ILE", "LEU," "MET", "VAL", "TRP","TYR"]
data["hydrophobe"]=data.type_aa.isin(hydrophobe)


#on garde que les résidues avec plus de 30% d'acessibilité au solvant pour la sphère

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

acces_solvant=[]
p = PDBParser()
structure = p.get_structure("structure","data/pdb/5jsi.pdb")
model = structure[0]
dssp = DSSP(model,"data/pdb/5jsi.pdb")

for i in range(len(list(dssp))):
	acces_solvant.append((list(dssp)[i][3]))

print(len(acces_solvant))


#data["acces_solvant"]=acces_solvant ## problème de taille ??
print(data)

 