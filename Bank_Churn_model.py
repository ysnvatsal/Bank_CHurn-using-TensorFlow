#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network

# ### Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


tf.__version__


# # Dataset Description:

# 
# This data set contains details of a bank's customers and the target variable is a binary variable reflecting the fact whether 
# the customer left the bank (closed his account) or he continues to be a customer.
# 
# 1) RowNumber                   : Sort Row Numbers from 1 to 10000
# 2) Customer_id                 : Sort Row Numbers from 1 to 10000 
# 3) Surname                     : Customer's last name
# 4) CreditScore                 : Credit score of the customer
# 5) Geography                   : The country from which the customer belongs
# 6) Gender                      : Male or Female
# 7) Tenure                      : Number of years for which the customer has been with the bank
# 8) Balance                     : Bank balance of the customer
# 9) NumOfProducts               : Number of bank products the customer is utilising
# 10) Age                        : Age of the customer
# 11) HasCrCard                  : Does he have a credit card?
# 12) IsActiveMember             : Is he using the card or the Bank?
# 13) EstimatedSalary            : How much salary estimated?
#     

# # Problem Statement:

# Find whether a customer leaves or continues to be with the bank

# ## Part 1 - Data Preprocessing

# ### Importing the dataset

# In[3]:


dataset = pd.read_csv('Churn_Modelling.csv')


# In[23]:


dataset.head(10)


# In[30]:


#Splitting into Training and Target variables
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# In[25]:


print(X)


# In[26]:


print(y)


# ### Encoding categorical data

# Label Encoding the "Gender" column. Neural Networks should contain only numerical data

# In[27]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])


# In[28]:


print(X)


# One Hot Encoding the "Geography" column

# In[8]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[9]:


print(X)


# ### Splitting the dataset into the Training set and Test set

# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ### Feature Scaling

# In[11]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Part 2 - Building the ANN

# ### Initializing the ANN

# In[13]:


ann = tf.keras.models.Sequential()


# ### Adding the input layer and the first hidden layer

# In[14]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# ### Adding the second hidden layer

# In[15]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# ### Adding the output layer

# In[16]:


#Note: Choosing activation function as sigmoid in the output layer not only gives us the prediction but it also generates the 
#probabilities of the customer that leaves or stays in the bank, If its a multiclass classification the activation 
#should be "softmax"

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ## Part 3 - Training the ANN

# ### Compiling the ANN

# In[17]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Training the ANN on the Training set

# In[18]:


ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


# ## Part 4 - Making the predictions and evaluating the model

# ### Predicting the result of a single observation

# 
# Use our ANN model Let's predict the customer with the following informations will leave the bank: 
# 
# Geography: France
# 
# Credit Score: 600
# 
# Gender: Male
# 
# Age: 40 years old
# 
# Tenure: 3 years
# 
# Balance: \$ 60000
# 
# Number of Products: 2
# 
# Does this customer have a credit card ? Yes
# 
# Is this customer an Active Member: Yes
# 
# Estimated Salary: \$ 50000
# 
# So, should we say goodbye to that customer ?

# **Solution**

# In[19]:


print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)


# Therefore, our ANN model predicts that this customer stays in the bank!
# 
# **Important note 1:** Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
# 
# **Important note 2:** Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.

# ### Predicting the Test set results

# In[20]:


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ### Making the Confusion Matrix

# In[39]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[58]:


# calling the roc_curve, extract the probability of 
# the positive class from the predicted probability
from sklearn.metrics import roc_curve,auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)

# AUC score that summarizes the ROC curve
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, lw = 2, label = 'ROC AUC: {:.2f}'.format(roc_auc))
plt.plot([0, 1], [0, 1],
         linestyle = '--',
         color = (0.6, 0.6, 0.6),
         label = 'random guessing')
plt.plot([0, 0, 1], [0, 1, 1],
         linestyle = ':',
         color = 'black', 
         label = 'perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc = "lower right")
plt.tight_layout()
plt.show()


# In[32]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)


# In[37]:


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


# In[53]:


precision, recall, thresholds = precision_recall_curve(
    y_test, y_pred, pos_label = 1)

# AUC score that summarizes the precision recall curve
avg_precision = average_precision_score(y_test, y_pred)

label = 'Precision Recall AUC: {:.2f}'.format(avg_precision)
plt.plot(recall, precision, lw = 2, label = label)
plt.xlabel('Recall')  
plt.ylabel('Precision')  
plt.title('Precision Recall Curve')
plt.legend()
plt.tight_layout()
plt.show()

