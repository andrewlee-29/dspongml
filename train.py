#%%
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv("pong_data.csv")
# print(data)
X = data.drop('paddle_y',axis=1) 
y = data['paddle_y']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=0)
# x = data.ball_y.to_numpy()
# y = data.paddle_y.to_numpy()
# X= x[:, np.newaxis]
# plt.scatter(X, y)
# plt.show()
# print(X)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
print(accuracy_score(ytest, y_model))
dump(model, 'mymodel.joblib') 
# %%
