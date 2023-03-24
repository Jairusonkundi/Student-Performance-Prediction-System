#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[17]:


# Aesthetic configuration for seaborn and matplotlib
plt.rcParams["font.family"] = "monospace"
sns.set_theme(style='darkgrid', palette='rocket')


# # Read Data

# In[18]:


df = pd.read_csv('Student_Performance_Dataset.csv')
df.head()


# In[19]:


df.shape


# In[20]:


df.info()


# # Numeric Data Statistics

# In[21]:


df.describe().T


# # Understanding the columns
# school - student’s school (binary: ‘GP’ - Gabriel Pereira or ‘MS’ - Mousinho da Silveira)
# 
# sex - student’s sex (binary: ‘F’ - female or ‘M’ - male)
# 
# age - student’s age (numeric: from 15 to 22)
# 
# address - student’s home address type (binary: ‘U’ - urban or ‘R’ - rural)
# 
# famsize - family size (binary: ‘LE3’ - less or equal to 3 or ‘GT3’ - greater than 3)
# 
# Pstatus - parent’s cohabitation status (binary: ‘T’ - living together or ‘A’ - apart)
# 
# Medu - mother’s education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
# 
# Fedu - father’s education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)
# 
# Mjob - mother’s job (nominal: ‘teacher’, ‘health’ care related, civil ‘services’ (e.g. administrative or police), ‘at_home’ or ‘other’)
# 
# Fjob - father’s job (nominal: ‘teacher’, ‘health’ care related, civil ‘services’ (e.g. administrative or police), ‘at_home’ or ‘other’)
# 
# reason - reason to choose this school (nominal: close to ‘home’, school ‘reputation’, ‘course’ preference or ‘other’)
# 
# guardian - student’s guardian (nominal: ‘mother’, ‘father’ or ‘other’)
# 
# traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# 
# studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# 
# failures - number of past class failures (numeric: n if 1<=n<3, else 4)
# 
# schoolsup - extra educational support (binary: yes or no)
# 
# famsup - family educational support (binary: yes or no)
# 
# paid - extra paid classes within the course subject (Portuguese) (binary: yes or no)
# 
# activities - extra-curricular activities (binary: yes or no)
# 
# nursery - attended nursery school (binary: yes or no)
# 
# higher - wants to take higher education (binary: yes or no)
# 
# internet - Internet access at home (binary: yes or no)
# 
# romantic - with a romantic relationship (binary: yes or no)
# 
# famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# 
# freetime - free time after school (numeric: from 1 - very low to 5 - very high)
# 
# goout - going out with friends (numeric: from 1 - very low to 5 - very high)
# 
# Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 
# Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 
# health - current health status (numeric: from 1 - very bad to 5 - very good)
# 
# absences - number of school absences (numeric: from 0 to 93)
# 
# G1 - first period grade (numeric: from 0 to 20)
# 
# G2 - second period grade (numeric: from 0 to 20)
# 
# G3 - final grade (numeric: from 0 to 20, output target)

# In[22]:


#Check null values
df.isnull().sum()


# # Exploratory Data Analysis (EDA)

# In[23]:


#Basic Analysis


# In[24]:


df['address'].value_counts()


# In[25]:


df['sex'].value_counts()


# In[26]:


df['famsize'].value_counts()


# In[27]:


df['internet'].value_counts()


# In[28]:


# correlation between Variables


# In[29]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)


# # Data Visualization

# In[30]:


#Categorical data visualization


# In[31]:


col_1=["school"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[32]:


col_1=["sex"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[33]:


col_1=["address"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[34]:


col_1=["famsize"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[35]:


col_1=["Pstatus"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[36]:


col_1=["Mjob"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[37]:


col_1=["Fjob"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[38]:


col_1=["reason"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[39]:


col_1=[ "guardian"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[40]:


col_1=["schoolsup"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[41]:


col_1=["famsup"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[42]:


col_1=[ "paid"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[43]:


col_1=["activities"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[44]:


col_1=["nursery"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[45]:


col_1=["internet"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[46]:


col_1=[ "romantic"]
def count_plot(col_1):
        plt.figure(figsize=(5,5))
        sns.countplot(x=col_1,data=df)

        plt.show()
        print(col_1,"\n",df[col_1].value_counts())
# Print
for i in col_1:
        count_plot(i)


# In[47]:


#Numeical data visualization


# In[48]:


col_2=["age", "Fedu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc" ,"health", "absences", "G1", "G2", "G3"]
def hist_plot(col_2):
        plt.figure(figsize=(5,5))
        sns.kdeplot(data=df, x=col_2)

        plt.show()
        print(col_2,"\n",df[col_2].value_counts())
# Print
for i in col_2:
        hist_plot(i)


# In[49]:


for i in col_2:
    ax=sns.boxplot(x=df[i])
    plt.show()


# In[50]:


sns.regplot(x="G1", y="G2", data=df)


# In[51]:


sns.regplot(x="G2", y="G3", data=df)


# In[52]:


sns.regplot(x="G1", y="G3", data=df)


# In[53]:


sns.regplot(x="Dalc",y="Walc",data=df)


# In[54]:


sns.regplot(x="health", y="G3", data=df)


# In[55]:


sns.regplot(x="Fedu", y="G3", data=df)


# In[56]:


# Create an overall health column for each entry
df['Overall Health'] = (0.5 * df['Dalc'] + 0.5 * df['Walc'] + 2 * df['health'] + df['famrel']) / 4


# # Overall Health vs. Grade
# Overall Health vs. Grade
# The relationship between these two features can be depicted using a simple regression plot.

# In[57]:


sns.regplot(x='Overall Health', y='G3', data=df)


#  # Absences vs. Grade

# In[58]:


sns.regplot(x='absences', y='G3', data=df).set(title='Absences vs G3')


# In[59]:


fig, axes = plt.subplots(2,1, figsize=(6,11))
sns.regplot(x='absences', y='G2', data=df, ax=axes[0])
axes[0].set(title='Absences vs G2')

sns.regplot(x='absences', y='G1', data=df, ax=axes[1])
axes[1].set(title='Absences vs G1')


# In[60]:


# This low correlation is because absent students (usually) revise the material missed,
#effectively accounting for their absence.


# In[61]:


#plot the correlation matrix to validate our hypothesis


# In[62]:


sns.heatmap(df[['absences', 'G1', 'G2', 'G3']].corr(), annot=True)


# # Age vs. Grade

# In[63]:


# Visualize distribution of `age`
sns.displot(x='age', data=df, kind='hist', kde=True)


# In[64]:


age_grade = df.groupby("age").aggregate({'G1': 'mean', 'G2': 'mean', 'G3': 'mean'})
age_grade.reset_index(inplace=True)
age_grade


# In[65]:


# As we can see we notice when age is older the grades got decreased.

 # We will plot the grades corresponding with ages.


# In[66]:


grades = ['G1', 'G2', 'G3']

for grade in grades:
    sns.barplot(data=age_grade, x='age', y=grade, palette='husl').set(xlabel='Age', ylabel=grade, title=f'Age vs. {grade}')
    plt.show()


# In[67]:


# The older a person is, the lower the grades they receive are; however, twenty-year-olds (in this dataset)
# exhibit an outstanding performance


# # Education Level vs. Job

# In[68]:


mjob_edu = df.groupby("Mjob").aggregate({"Medu": "mean"})
mjob_edu.reset_index(inplace=True)
mjob_edu.sort_values(by='Medu', ascending=False, inplace=True)


# In[69]:


sns.barplot(x='Mjob', y='Medu', data=mjob_edu).set(xlabel='Job', ylabel='Education Lvl.', title='Job vs. Edu. Lvl. (Mother)')


# In[70]:


# As expected, teachers and health care professionals need to have a high education level in order
# to acquire a job in the industry,and conversely with at home mother.


# # Fathers' Job and Education

# In[71]:


fjob_edu = df.groupby("Fjob").aggregate({"Fedu": "mean"})
fjob_edu.reset_index(inplace=True)
fjob_edu.sort_values(by='Fedu', ascending=False, inplace=True)


# In[72]:


sns.barplot(x='Fjob', y='Fedu', data=fjob_edu).set(xlabel='Job', ylabel='Education Lvl.', title='Job vs. Edu. Lvl. (Father)')


# # Time Productivity vs. Grade

# In[73]:


df['Time Productivity'] = 0.5 * df['traveltime'] + 2 * df['studytime']
sns.regplot(x='Time Productivity', y='G3', data=df)


# In[74]:


# As expected, students with a greater time productivity have better grades.


# In[75]:


sns.regplot(x='studytime', y='G3', data=df)


# In[76]:


sns.regplot(x='traveltime', y='G3', data=df)


# In[77]:


#Conclusion
#This section confirmed the obvious: students who study more receive better grades whereas
#students who travel more or study less receive lower grades.


# In[78]:


# Job vs. Grade


# In[79]:


mjob_gr = df.groupby("Mjob").aggregate({"G3": "mean"}).reset_index()
mjob_gr


# In[80]:


sns.barplot(data=mjob_gr, x='Mjob', y='G3', palette='husl').set(xlabel='Job', ylabel='G3', title='Job vs. Grade (Mother)')


# In[81]:


# Fathers' Job


# In[82]:


fjob_gr = df.groupby("Fjob").aggregate({"G3": "mean"}).reset_index()
fjob_gr


# In[83]:


sns.barplot(data=fjob_gr, x='Fjob', y='G3', palette='husl').set(xlabel='Job', ylabel='G3', title='Job vs. Grade (Father)')


# In[84]:


# Family Size vs. Grade


# In[85]:


size_gr = df.groupby("famsize").aggregate({"G1": "mean", "G2": "mean", "G3": "mean"}).reset_index()
size_gr


# In[86]:


for grade in grades:
    sns.barplot(data=size_gr, x='famsize', y=grade, palette='husl').set(xlabel='Family Size', ylabel=grade, title=f'Family Size vs. {grade}')
    plt.show()


# In[87]:


# As the barplots show, children with no siblings tend to score slightly 
# indeed, very slightly - higher than students with siblings


# In[88]:


# Activites vs. Grade


# In[89]:


act_gr = df.groupby("activities").aggregate({"G1": "mean", "G2": "mean", "G3": "mean"}).reset_index()
act_gr


# In[90]:


for grade in grades:
    sns.barplot(data=act_gr, x='activities', y=grade, palette='husl')
    plt.show()


# In[91]:


out_gr = df.groupby("goout").aggregate({"G1": "mean", "G2": "mean", "G3": "mean"}).reset_index()
out_gr


# In[92]:


for grade in grades:
    sns.barplot(data=out_gr, x='goout', y=grade, palette='rocket')
    plt.show()


# # Data Pre-Processing
# This involves a number of activities such as:
# 
# Assigning numerical values to categorical data; Handling missing values; and Normalizing the features (so that features on small scales do not dominate when fitting a model to the data).

# In[93]:


#list of columns that are categorical
cat_col = df.select_dtypes(include=['object']).columns.tolist()
#list of columns that are numerical
num_col = df.select_dtypes(include=['number']).columns.tolist()
cat_col


# In[94]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,MinMaxScaler


# # Label encoding
# Here, I assign the 32 features to a NumPy array X and encoded the original string representation on the categorical columns into integers to start the machine learning phase

# # label encoding
# One-hot encoding  technique is used to convert categorical variables into numerical variables 
# that can be used in machine learning model. It creates a new binary column for each unique value 
# in the categorical variable(s). The binary column is set to 1 if the original column had that value
# for that row, and 0 otherwise.
# The get_dummies() function from the pandas library is used to perform one-hot encoding

# In[95]:


new_df = pd.get_dummies(df, columns=['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason',
                                     'guardian','schoolsup','famsup','paid','activities','nursery','higher','internet',
 'romantic'])


# In[96]:


df.head()


# In[97]:


# We drop G3 column


# In[139]:


X = new_df.drop('G3', axis=1).values
y = new_df['G3'].values


# # spliting the data into train and test set

# In[140]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[141]:


# Scaling the data using pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)
X_test


# # Training data using different models

# # Linear Regression

# In[142]:


from sklearn.linear_model import LinearRegression

# initializing the algorithm
lin_reg = LinearRegression(normalize=True)

# Fitting Simple Linear Regression to the Training set
lin_reg.fit(X_train,y_train)


# In[143]:


from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square


# In[144]:


test_pred = lin_reg.predict(X_test)
train_pred = lin_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])


#  # SGDRegressor

# In[145]:


from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)
sgd_reg.fit(X_train, y_train)

test_pred = sgd_reg.predict(X_test)
train_pred = sgd_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Stochastic Gradient Descent", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)


# # Ridge Regression

# In[146]:


from sklearn.linear_model import Ridge

model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, test_pred) , cross_val(Ridge())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# # RandomForestRegressor

# In[147]:


from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=1000)
rf_reg.fit(X_train, y_train)

test_pred = rf_reg.predict(X_test)
train_pred = rf_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Random Forest Regressor", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)


# # Support Vector Machine

# In[148]:


from sklearn.svm import SVR

svm_reg = SVR(kernel='rbf', C=1000000, epsilon=0.001)
svm_reg.fit(X_train, y_train)

test_pred = svm_reg.predict(X_test)
train_pred = svm_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["SVM Regressor", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)


# # K Nearest Neighbors

# In[149]:


from sklearn.neighbors import KNeighborsClassifier

knn_reg = SVR(kernel='rbf', C=1000000, epsilon=0.001)
knn_reg.fit(X_train, y_train)

test_pred = knn_reg.predict(X_test)
train_pred = knn_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["KNeighborsClassifier", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)


# # ACCCURACY

# In[150]:


results_df


# In[151]:


results_df.set_index('Model', inplace=True)
results_df['R2 Square'].plot(kind='barh', figsize=(12, 8))


# # Feature Selection

# In[166]:


X = new_df.drop('G3', axis=1)
y = new_df['G3']


# In[ ]:





# In[167]:


# Important feature using ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)


# In[168]:


print(selection.feature_importances_)


# In[169]:


plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# In[206]:


X = df[[ 'studytime','G1', 'G2']]
y = df[['G3']]


# In[208]:


sns.heatmap(df[['studytime','G1', 'G2','G3']].corr(), annot=True)


# In[209]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[215]:


from sklearn.linear_model import LinearRegression

# initializing the algorithm
lin_regress = LinearRegression(normalize=True)

# Fitting Simple Linear Regression to the Training set
lin_regress.fit(X_train,y_train)


# In[ ]:





# In[216]:


test_pred = lin_regress.predict(X_test)
train_pred = lin_regress.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])


# In[217]:


from sklearn.linear_model import Ridge

model = Ridge(alpha=100, solver='cholesky', tol=0.0001, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, test_pred) , cross_val(Ridge())]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# In[ ]:





# In[218]:


from sklearn.linear_model import SGDRegressor

sgd_regres = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)
sgd_regres.fit(X_train, y_train)

test_pred = sgd_regres.predict(X_test)
train_pred = sgd_regres.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)
print('====================================')
print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Stochastic Gradient Descent", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)


# In[220]:


from sklearn.ensemble import RandomForestRegressor

rfres = RandomForestRegressor(n_estimators=1000)
rfres.fit(X_train, y_train)

test_pred = rfres.predict(X_test)
train_pred = rfres.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

results_df_2 = pd.DataFrame(data=[["Random Forest Regressor", *evaluate(y_test, test_pred), 0]], 
                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])
results_df = results_df.append(results_df_2, ignore_index=True)


# In[ ]:





# In[ ]:





# 
# # ACCURACY

# In[221]:


results_df


# In[222]:


results_df.set_index('Model', inplace=True)
results_df['R2 Square'].plot(kind='barh', figsize=(12, 8))


# # Saving the model

# In[223]:


from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
print(y_pred)


# In[224]:


import joblib


# In[225]:


joblib.dump(rfr,r'C:\Users\Jairo\Desktop\Python\Student Performance Prediction System\models\rfr.sav')


# In[ ]:





# In[226]:


from sklearn.linear_model import LinearRegression

lin_regress= LinearRegression()
lin_regress.fit(X_train, y_train)
y_pred = lin_regres.predict(X_test)
print(y_pred)


# In[204]:


import joblib


# In[227]:


joblib.dump(lin_regress,r'C:\Users\Jairo\Desktop\Python\Student Performance Prediction System\models\lin_regress.sav')


# In[ ]:




