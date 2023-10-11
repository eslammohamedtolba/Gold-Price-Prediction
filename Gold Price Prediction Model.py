# import requied modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# uploading gold price dataset
gold_price_dataset = pd.read_csv("gld_price_data.csv")
# show the dataset
gold_price_dataset.head()
# # show dataset shape
gold_price_dataset.shape
# # show some statistical info about dataset
gold_price_dataset.describe()
# # check if there is any none values in the dataset to make data cleaning or not
gold_price_dataset.isnull().sum()



plt.figure(figsize=(8,8))
correlation_values = gold_price_dataset.corr()
sns.heatmap(correlation_values,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap="Blues")
# checking the distribution of the gold price
sns.distplot(gold_price_dataset['GLD'],color='red')



# split data into input and label data
X = gold_price_dataset.drop(columns=['GLD','Date'],axis=1)
Y = gold_price_dataset['GLD']
print(X)
print(Y)
# split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)



# create the model and train it
RFModel = RandomForestRegressor()
RFModel.fit(x_train,y_train)
# make the model predict the train data
predicted_train_data = RFModel.predict(x_train)
# make the model predict the test data
predicted_test_data = RFModel.predict(x_test)
accuracy_pridcted_train_values = r2_score(predicted_train_data,y_train)
accuracy_pridcted_test_values = r2_score(predicted_test_data,y_test)
print(accuracy_pridcted_train_values,accuracy_pridcted_test_values)



# plot the difference between the predicted and actual values
plt.title("the predicted and actual values on train data")
plt.xlabel("actaul values")
plt.ylabel("predicted values")
plt.scatter(y_train,predicted_train_data,marker="X",color="blue")
plt.title("the predicted and actual values on test data")
plt.xlabel("actaul values")
plt.ylabel("predicted values")
plt.scatter(y_test,predicted_test_data,marker="^",color="red")
plt.show()

