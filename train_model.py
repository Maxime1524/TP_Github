def train_model(train_X, train_y, test_X, model):
    """entraine un modele ML de classification et retourne des predictions sur
    le dataset de test"""
    model.fit(train_X,train_y) 
    prediction=model.predict(test_X)
    return prediction


def get_learning_curves(model, X, y):
    """Calcule les courbes d'apprentissage du modèle"""
    from sklearn.model_selection import learning_curve
    import numpy as np
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    return train_sizes, train_mean, train_std, val_mean, val_std

"Modifications de l'Étudiant A :"
"""- Ajout de la fonction get_learning_curves() dans train_model.py
 Cette fonction calcule les courbes d'apprentissage du modèle """