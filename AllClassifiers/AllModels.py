#importing libraries
import pandas as pd
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score

#loading preprocessed dataset
cancer = pd.read_csv(sys.argv[1], header = None)

#features set
X = np.array(cancer.ix[:,0:28].values)
#ouoput label
y = np.array(cancer.ix[:,29].values)
#creating empty dictionary to append all classifiers
dictOfClassifiers={}
knn=KNeighborsClassifier(n_neighbors = 7,weights='uniform')
dictOfClassifiers.update({ knn : "K nearest neighbour"})
svm = SVC(kernel = "rbf", C=1)
dictOfClassifiers.update({svm:"SVM"})
lr = LogisticRegression(C=1, max_iter = 100)
dictOfClassifiers.update({lr:"Logistic Regression"})
dt = DecisionTreeClassifier(criterion='entropy', max_depth=4)
dictOfClassifiers.update({dt:"Decision Tree"})
nb = GaussianNB()
dictOfClassifiers.update({nb:"Naive Bayes"})
p =Perceptron( eta0=1, alpha = 0.0001)
dictOfClassifiers.update({p:"Perceptron"})
ann = MLPClassifier(hidden_layer_sizes=(3, 5), activation='relu', alpha=1, max_iter=2000, learning_rate='adaptive')
dictOfClassifiers.update({ann:"Neural Network"})
dnn = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', alpha=1, max_iter=2000, learning_rate='adaptive')
dictOfClassifiers.update({dnn:"Deep Learning"})
rfc=RandomForestClassifier(max_depth= 14, n_estimators= 19,criterion='entropy')
dictOfClassifiers.update({rfc:"Random Forest"})
abc=AdaBoostClassifier(n_estimators = 50)
dictOfClassifiers.update({abc:"Aadaboosting"})
gbc=GradientBoostingClassifier(n_estimators = 190)
dictOfClassifiers.update({gbc:"Gradient Boosting"})
bag=BaggingClassifier(max_features= 18, n_estimators= 23)
dictOfClassifiers.update({bag:"Bagging"})
#testing all classifiers using 10 fold cross validation
for key in dictOfClassifiers.keys():
    scores = cross_val_score(key, X, y, cv=10, scoring='accuracy')
    scoreF1 = cross_val_score(key, X, y, cv=10, scoring='f1')
    print("Model :"+str(dictOfClassifiers[key])+" Accuracy : "+str(np.mean(scores)))
    print("Model :" + str(dictOfClassifiers[key]) + " F1_score : " + str(np.mean(scoreF1)))
print("Done")