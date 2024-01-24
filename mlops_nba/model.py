import os
from pathlib import Path
from joblib import dump
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_and_save_model(players_df):
    # Définir le préprocesseur
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']),
            ('cat', OneHotEncoder(handle_unknown='error'), ['Pos', 'Tm'])  # Changé à 'error'
        ])

    # Définir le modèle
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    # Séparer les données en ensembles d'entraînement et de test
    X = players_df.drop(['Player', 'PTS', 'FG%'], axis=1)
    y_pts = players_df['PTS']
    y_fg = players_df['FG%']
    X_train_pts, X_test_pts, y_train_pts, y_test_pts = train_test_split(X, y_pts, test_size=0.2, random_state=42)
    X_train_fg, X_test_fg, y_train_fg, y_test_fg = train_test_split(X, y_fg, test_size=0.2, random_state=42)

    # Entraîner le modèle pour prédire PTS
    model.fit(X_train_pts, y_train_pts)
    pts_preds = model.predict(X_test_pts)
    print(f'RMSE for PTS prediction: {mean_squared_error(y_test_pts, pts_preds, squared=False)}')

    # Entraîner le modèle pour prédire FG%
    model.fit(X_train_fg, y_train_fg)
    fg_preds = model.predict(X_test_fg)
    print(f'RMSE for FG% prediction: {mean_squared_error(y_test_fg, fg_preds, squared=False)}')

    # Compter les fichiers dans le dossier 'Model'
    model_dir = Path('C:\\Users\\loicc\\OneDrive - Efrei\\Bureau\\COURS\\M2\\S9\\Machine Learning in Production\\Data Pipeline\\mlops-nba\\Model')
    model_count = len(list(model_dir.glob('*.joblib'))) + 1

    # Sauvegarder le modèle
    model_file_name = model_dir / f'model_{model_count}.joblib'
    dump(model, model_file_name)
    print(f'Modèle sauvegardé sous : {model_file_name}')