import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP


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

    data = pd.DataFrame(data=data_pdb, columns=[
                        "position", "type_aa", "x", "y", "z"])

    return data


def recup_acc_solvant(fichier_pdb):
    """
    entrée le fichier pdb
    utilise Biopython et le programme dssp 
    sortie une liste de l'accesibilité relative (entre 0 et 1) des résidues
   
    """
    acces_solvant = []
    p = PDBParser()
    id_structure = fichier_pdb.split(".")[0]

    structure = p.get_structure(id_structure, fichier_pdb)
    model = structure[0]
    dssp = DSSP(model, fichier_pdb, dssp='mkdssp')

    for i in range(len(list(dssp))):
        # ajout de chaque élément arrondi au millième
        acces_solvant.append(round((list(dssp)[i][3]), 3))

    return acces_solvant
