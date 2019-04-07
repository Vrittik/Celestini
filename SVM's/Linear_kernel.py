import pandas as pd
import numpy
datset=pd.read_csv("zoo.data")
X=datset.iloc[:,1:17].values
y=datset.iloc[:,17].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

classifier.score(X_test,y_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score
#

scores = cross_val_score(classifier, X_train, y_train, cv=5)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.model_selection import KFold
kf=KFold(n_splits=5)
score_linear_kernel=[]
def get_score(model,X_train,X_test,y_train,y_test):
    model.fit(X_train,y_train)
    return model.score(X_test,y_test)
score=get_score(classifier,X_train,X_test,y_train,y_test)

for train_index,test_index in kf.split(X_train,y_train):
    x_train,x_test=X_train[train_index],X_train[test_index]
    Y_train,Y_test=y_train[train_index],y_train[test_index]
    score_linear_kernel.append(get_score(classifier,x_train,x_test,Y_train,Y_test))
print("After k fold for k=5 in linear kernel, the scores array is as follows:")
print(score_linear_kernel)
 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

