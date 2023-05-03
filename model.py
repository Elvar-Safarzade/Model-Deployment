import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings

data=pd.read_csv(r"C:\Users\Lenovo\Desktop\Machine Learning\student-mat.csv")
print(data.head())

print(data.shape)

print(data.describe())

print(data.info())
print(data.corr())

df=data[['Dalc','Walc','G1','G2','G3']]
print(df.head())

X=df.drop('G3',axis=1)
y=df['G3']

sns.lmplot(x='Dalc',y='G3',data=df)
plt.show()

sns.lmplot(x='Walc',y='G3',data=df)
plt.show()

sns.lmplot(x='G1',y='G3',data=df)
plt.show()

sns.lmplot(x='G2',y='G3',data=df)
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=41)

from sklearn.linear_model import LinearRegression
model =LinearRegression()

model.fit(X_train,y_train)

print(model.score(X_train,y_train))

prediction_test=model.predict(X_test)
print(y_test,prediction_test)


print("Mean sq.error between y_test and prediction_test",np.mean(prediction_test-y_test)**2)


import pickle 

pickle.dump(model, open('model.pkl', 'wb'))

model=pickle.load(open('model.pkl','rb'))

print(model.predict([[4,7,4,11]]))












