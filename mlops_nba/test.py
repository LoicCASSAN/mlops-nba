# Data Processing
import pandas as pd
import numpy as np
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
RAW_DATA_DIR = Path('C:\\Users\\loicc\\OneDrive - Efrei\\Bureau\\COURS\\M2\\S9\\Machine Learning in Production\\Data Pipeline\\mlops-nba')

# Remplacez 'nom_du_fichier.csv' par le nom de votre fichier CSV
filename = RAW_DATA_DIR / '2023-2024 NBA Player Stats - Regular.csv'
print(f"Running on file: {filename}")
players = pd.read_csv(filename, sep=',', encoding='Windows-1252')
# Spécifiez le chemin de destination pour le fichier Parquet
destination_dir = Path('C:\\Users\\loicc\\OneDrive - Efrei\\Bureau\\COURS\\M2\\S9\\Machine Learning in Production\\Data Pipeline\\mlops-nba\\stockage')

# Assurez-vous que le dossier de destination existe
destination_dir.mkdir(parents=True, exist_ok=True)

# Utilisez la méthode to_parquet pour stocker le DataFrame au format Parquet
destination_file = destination_dir / 'players_raw.parquet'
players.to_parquet(destination_file, index=False)

print(f"DataFrame saved as Parquet: {destination_file}")
players.sort_values(by=['Player'], ascending=True).head(5)
assert sum(players.isnull().sum()) == 0, "There are not null values in the dataset"
players["EFF"] = players.PTS + players.TRB + players.AST + players.STL + players.BLK - (players.FGA - players.FG) - (players.FTA - players.FT) - players.TOV
ages = players.Age.describe().round(decimals=1) # used to specify the first 25%, defining what is a young player
points = players.PTS.describe().round(decimals=1)
# With the graph below, we can see that within <23y (what we have defined to be a young age), if we have more than 15 points we are special. 
# Those data will then be used to filter the current base player and keep only special ones.

young_age = ages["25%"]
futur_super_star_def = f"(EFF >= 12) & (PTS >= 15) & (Age <= {young_age})"
players.query(futur_super_star_def).sort_values("EFF", ascending=False).sort_values(["Age", "EFF"], ascending=True)
players['TS%'] = np.where((2 * (players['FGA'] + 0.44 * players['FTA'])) != 0, players['PTS'] / (2 * (players['FGA'] + 0.44 * players['FTA'])), 0)
players["position"] = players.Pos.map({"PG": "Backcourt", "SG": "Backcourt", "SF": "Wing", "SF-PF": "Wing", "PF": "Big", "C": "Big", })
# Spécifiez le chemin de destination pour le fichier Parquet
destination_dir = Path('C:\\Users\\loicc\\OneDrive - Efrei\\Bureau\\COURS\\M2\\S9\\Machine Learning in Production\\Data Pipeline\\mlops-nba\\stockage')

# Assurez-vous que le dossier de destination existe
destination_dir.mkdir(parents=True, exist_ok=True)

# Utilisez la méthode to_parquet pour stocker le DataFrame au format Parquet
destination_file = destination_dir / 'players_final.parquet'
players.to_parquet(destination_file, index=False)

print(f"DataFrame saved as Parquet: {destination_file}")

# API request
