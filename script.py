#%%
# 
# Ouvrir le fichier CSV d'entrée en mode lecture
with open('save_score_challenge.csv', 'r') as file_in:
    # Lire toutes les lignes du fichier
    lines = file_in.readlines()

# Ouvrir le fichier CSV de sortie en mode écriture
with open('save_score_challenge_updated.csv', 'w') as file_out:
    # Parcourir chaque ligne
    for line in lines:
        # Supprimer les caractères de fin de ligne (\n)
        line = line.rstrip('\n')
        # Vérifier si la ligne se termine déjà par une virgule
        # line = line[:-6]
        line += ',0.2,0'
        # Écrire la ligne mise à jour dans le fichier de sortie
        file_out.write(line + '\n')

print("Fichier mis à jour avec succès.")

# %%
