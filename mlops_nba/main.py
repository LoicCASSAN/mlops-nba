import pandas as pd
from pathlib import Path
from processing import process_players_data, save_dataframe_as_parquet, process_and_train
import time

if __name__ == "__main__":
    # Exécuter une fois au début
    loaded_files = {}
    RAW_DATA_DIR = Path('C:\\Users\\loicc\\OneDrive - Efrei\\Bureau\\COURS\\M2\\S9\\Machine Learning in Production\\Data Pipeline\\mlops-nba')

    # Lire le fichier CSV initial
    filename = RAW_DATA_DIR / '2023-2024 NBA Player Stats - Regular.csv'
    print(f"Running on file: {filename}")
    players_df = pd.read_csv(filename, sep=',', encoding='Windows-1252')
    players_df['filename'] = filename.name

    # Traitement des données des joueurs
    players_df = process_players_data(players_df)

    # Sauvegarder le DataFrame final
    save_dataframe_as_parquet(players_df, 'C:\\Users\\loicc\\OneDrive - Efrei\\Bureau\\COURS\\M2\\S9\\Machine Learning in Production\\Data Pipeline\\mlops-nba\\stockage', 'players_final.parquet', 1)

    # Traitement et entraînement supplémentaires
    process_and_train(loaded_files, players_df)

    # # Boucle infinie pour garder le script en cours d'exécution
    # try:
    #     while True:
    #         # Ici, vous pouvez ajouter des tâches à exécuter continuellement
    #         # par exemple, vérifier de nouveaux fichiers toutes les N secondes
    #         time.sleep(10)  # Attendre 10 secondes entre chaque vérification
    # except KeyboardInterrupt:
    #     print("Arrêt du script main.py.")



    try:
        while True:
            # Définir le chemin vers le dossier à vérifier
            append_data_dir = Path('C:\\Users\\loicc\\OneDrive - Efrei\\Bureau\\COURS\\M2\\S9\\Machine Learning in Production\\Data Pipeline\\mlops-nba\\Append Data')

            # Vérifier les nouveaux fichiers
            new_files_detected = False
            for file_path in append_data_dir.glob('*.csv'):
                if file_path.name not in loaded_files:
                    new_files_detected = True
                    loaded_files[file_path.name] = True  # Marquer le fichier comme lu

            # Exécuter process_and_train si de nouveaux fichiers ont été détectés
            if new_files_detected:
                print("Nouveaux fichiers détectés. Traitement en cours...")
                process_and_train(loaded_files, players_df)

            # Attendre avant la prochaine vérification
            time.sleep(1)  # Vérifier toutes les secondes
    except KeyboardInterrupt:
        print("Arrêt du script main.py.")