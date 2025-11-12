import pandas as pd 
from sklearn import svm  
from train_model import train_model, get_learning_curves
from preprocess_data import preprocess_data
import matplotlib.pyplot as plt

iris = pd.read_csv("InputData/Iris.csv") #load the dataset

test_size = 0.15 # Nouvelle modif Étudiant A - 15% pour test				#into 70% for train and 30% for test


train, test = preprocess_data(iris, test_size)
# training data features
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
# target of our training data
train_y = train.Species
# test data features
test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
#target value of test data
test_y = test.Species   


model = svm.SVC()

# Calculer les courbes d'apprentissage AVANT l'entraînement final
print("Calcul des courbes d'apprentissage...")
train_sizes, train_mean, train_std, val_mean, val_std = get_learning_curves(model, train_X, train_y)

# Afficher les courbes d'entraînement
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Score d\'entraînement')
plt.plot(train_sizes, val_mean, 'o-', color='g', label='Score de validation')

# Ajouter les zones d'écart-type
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')

plt.xlabel('Nombre d\'exemples d\'entraînement')
plt.ylabel('Score (Accuracy)')
plt.title('Courbes d\'apprentissage - SVM sur Iris')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig('courbes_entrainement.png')
print("Courbes sauvegardées dans 'courbes_entrainement.png'")
plt.show()

# Entraînement final et prédiction
prediction, trained_model = train_model(train_X, train_y, test_X, model)
print(f"\nPrédictions effectuées sur {len(prediction)} échantillons de test")