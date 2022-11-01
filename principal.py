
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd #manipulacao de dados

x= np.array([[1,2],[1,2], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,2], [-2,2], [2,7], [-4,1], [0,0]])
y = np.array([15, 15, 15, 3, 4, 3, 3, 15, 3, 4, 4, 7])


# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#Cria um classificador Gaussiano
model = GaussianNB()

#Treina o modelo usando os dados de treino 
model.fit(X_train, y_train)


#Resultado de previsão 
teste = np.array([ [1,2],[0,0],[-2,0],[2,7] ])
#predicted = model.predict([[1,2],[0,0]]) #fazer a previsão em cima desses 2 numeros
predicted = model.predict(teste) #fazer a previsão em cima desse array teste
print(predicted)

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, predicted)*100)
