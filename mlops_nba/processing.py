import pandas as pd
import numpy as np
import os
from pathlib import Path
from model import train_and_save_model

def save_dataframe_as_parquet(dataframe, path, filename, file_count=None):
    if file_count and file_count > 0:
        # Modifier le nom du fichier pour inclure le nombre de fichiers traités
        modified_filename = filename.replace('.parquet', f'_{file_count}.parquet')
    else:
        # Garder le nom de fichier original si aucun fichier n'a été traité
        modified_filename = filename

    # Chemin complet du fichier
    full_path = os.path.join(path, modified_filename)

    # Sauvegarder le DataFrame en fichier Parquet
    dataframe.to_parquet(full_path, index=False)
    print(f"DataFrame sauvegardé en tant que fichier Parquet : {full_path}")

def process_players_data(players):
    # Assurer que le DataFrame n'est pas vide
    if players.empty:
        raise ValueError("Le DataFrame 'players' est vide.")

    # Ajouter la colonne EFF
    players["EFF"] = players.PTS + players.TRB + players.AST + players.STL + players.BLK - \
                     (players.FGA - players.FG) - (players.FTA - players.FT) - players.TOV

    # Calculer les statistiques d'âge et de points
    ages = players.Age.describe().round(decimals=1)
    points = players.PTS.describe().round(decimals=1)

    # Définir les critères pour les futurs superstars
    young_age = ages["25%"]
    futur_super_star_def = f"(EFF >= 12) & (PTS >= 15) & (Age <= {young_age})"
    players.query(futur_super_star_def).sort_values("EFF", ascending=False).sort_values(["Age", "EFF"], ascending=True)

    # Calculer le pourcentage de tir
    players['TS%'] = np.where((2 * (players['FGA'] + 0.44 * players['FTA'])) != 0, 
                              players['PTS'] / (2 * (players['FGA'] + 0.44 * players['FTA'])), 0)

    players["Pos"] = players["Pos"].replace("C-PF", "C")
    # Mapper les positions des joueurs
    players["position"] = players.Pos.map({"PG": "Backcourt", "SG": "Backcourt", 
                                           "SF": "Wing", "SF-PF": "Wing", 
                                           "PF": "Big", "C": "Big"})

    # Retourner le DataFrame traité
    return players

def process_and_train(loaded_files, players_df):
    RAW_DATA_DIR = Path('C:\\Users\\loicc\\OneDrive - Efrei\\Bureau\\COURS\\M2\\S9\\Machine Learning in Production\\Data Pipeline\\mlops-nba\\Append Data')

    # Parcourir tous les fichiers CSV dans le dossier
    for file_path in RAW_DATA_DIR.glob('*.csv'):
        file_name = file_path.name

        # Vérifier si le fichier a déjà été lu
        if file_name not in loaded_files:
            # Lire les données du fichier CSV
            new_data = pd.read_csv(file_path, sep=';', encoding='Windows-1252')
            new_data['filename'] = file_name

            # Sauvegarder la taille actuelle du DataFrame
            current_length = len(players_df)

            # Concaténer avec le DataFrame principal
            players_df = pd.concat([players_df, new_data], ignore_index=True)

            # Calculer le nombre de lignes ajoutées
            new_lines_added = len(players_df) - current_length
            if new_lines_added > 0:
                print(f"{new_lines_added} nouvelles lignes ajoutées depuis {file_name}.")

            # Marquer le fichier comme lu
            loaded_files[file_name] = True

    # Sauvegarder le DataFrame en tant que fichier Parquet
    save_dataframe_as_parquet(players_df, 'C:\\Users\\loicc\\OneDrive - Efrei\\Bureau\\COURS\\M2\\S9\\Machine Learning in Production\\Data Pipeline\\mlops-nba\\stockage', 'players_append.parquet', len(loaded_files))
    
    # Entraîner et sauvegarder le modèle
    train_and_save_model(players_df)
