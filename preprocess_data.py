from sklearn.model_selection import train_test_split

def preprocess_data(data, testSize):
    """Nettoie, met en forme les données et prépare les ensembles de train et 
    de test"""
    # Supprimer la colonne 'Id' qui n'est pas utile pour l'entraînement
    if 'Id' in data.columns:
        data = data.drop('Id', axis=1)
    
    train, test = train_test_split(data, test_size = testSize)
    return train, test