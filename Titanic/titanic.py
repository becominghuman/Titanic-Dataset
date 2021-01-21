import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer

train = pd.read_csv("/Users/fadouabadr/Library/Mobile Documents/com~apple~CloudDocs/OBJECTIF CDI/IT stuff/PYTHON/Titanic/Datas/Train.csv")
test = pd.read_csv("/Users/fadouabadr/Library/Mobile Documents/com~apple~CloudDocs/OBJECTIF CDI/IT stuff/PYTHON/Titanic/Datas/Test.csv")

train_copy = train.copy()
test_copy = test.copy()

survived_train = train['Survived'].copy() # Serues object
PassengerId_test = test['PassengerId']


# ***************************************************************************************************
# ***************************************************************************************************
#print first 3 rows of train and test data
train.head(3)
test.head(3)

#print the total number of columns in the train data
print("The total number of rows in the training data is", format(train.shape[0]))
print("the total number of rows in the test data is", format(test.shape[0]))
print(60*'-')
print("The total number of columns in the training data is", format(train.shape[1]))
print("the total number of columns in the test data is", format(test.shape[1]))

print("The different attributes in the train set are", format(train.columns))

#let's remove the PassengerId/Cabin attribute from the test and train set as they are literally useless for our models
train = train.drop(columns=['PassengerId', 'Cabin'], axis=1)
test = test.drop(columns=['PassengerId','Cabin'], axis=1)
# ***************************************************************************************************
# ***************************************************************************************************




# *************************************             ******************************************
# ==================================== DATA ANALYSIS ==========================================
# *************************************             ******************************************
print((train.groupby(['Sex','Survived']).Survived.count() * 100) / train.groupby('Sex').Survived.count())

print((train.groupby(['Pclass','Survived']).Survived.count() * 100) / train.groupby('Pclass').Survived.count())

print((train.groupby(['Embarked','Survived']).Survived.count() * 100) / train.groupby('Embarked').Survived.count())

print(train.groupby(by=["Survived"]).mean()["Age"])


#let's use the info() method to get a better view of our train set 
info_train = train.info()
info_test = test.info()




# ************************************                                 ******************************************
# ==================================== DEALING WITH NAN/MISSING VALUES ==========================================
# ************************************                                 ******************************************

datas = [train,test]

#train dataset : 177 values are missing in the 'Age' attribute, 2 are missing in the 'Embarked' attribute
#test dataset : 86 values are missing in the 'Age' attrivute, 1 is missing in the 'Fare' attribute
#Let's visualise the number of null-values/missing values in the training set using matplotlib's bar() method
for data in datas:
    print(data.isna().sum()) #here isna and isnull is the same (nan values)
    plt.figure(figsize=(10,3))
    plt.bar(data.columns, data.isnull().sum())
    plt.xticks(rotation=90)
    plt.xlabel("Attributes"); plt.ylabel("# of missing values in the dataset")
    print(20 * '*')

print('Survived/Dead\n', train['Survived'].value_counts())
print(50*'-')
#print('Class\n', train["Pclass"].value_counts())
print(50*'-')
print('Sex\n', train["Sex"].value_counts())
print(50*'-')
print('Siblings/Spouses\n', train['SibSp'].value_counts())
print(50*'-')
print('Parents/Children\n', train['Parch'].value_counts())
print(50*'-')
print('Embarked\n', train['Embarked'].value_counts())
















# ***************************************************************************************************
# ***************************************************************************************************
#*******************************************************************************************************
#The 'Pclass' attribute does not have any missing values

#----------------------------------------------------- Plot ----------------------------------------------------------------------
# Visualising whether Pclass affects the survival rate or not
plt.figure(figsize=(10,5))
sns.countplot(x = 'Pclass', hue = 'Survived', data = train, palette = 'Paired')
plt.show()
# We notice that the 3rd class is the class that had the most deadly rate followed by the 2dn class and the 1st. 
# This will be confirmed when looking to the Correlation matrix

#Let's plot several conditional relationship related the the Pclass attributes
grid = sns.FacetGrid(train, col='Survived', row='Pclass', palette='Paired')
grid.map_dataframe(sns.histplot, x='Age')
plt.legend()
plt.show()

fg = sns.FacetGrid(train, hue='Pclass', aspect=3)
fg.map(sns.kdeplot, 'Age', shade=True)
plt.legend()
fg.set(xlim=(0, 80))

#----------------------------------------------------- LabelBinarizer ----------------------------------------------------------------------
encoder = LabelBinarizer()
train_pclass = encoder.fit_transform(train["Pclass"])
test_pclass = encoder.fit_transform(test["Pclass"])

#use of the classes_ attribute to see which index corresponds to which class
#Rq: do not confuse the class (which is an indication of the socio economical background) and the class (which is .....)
print('The different classes in the Pclass attribute are', encoder.classes_) #class 0 = 1st class / class 1 = 2nd Class / class 2 = 3rd Class

#concatenate with the new dummies variables and delete the 'Embarked' attribute
#Rq: the train_pclass output is an array and we have to convert it to a df
train.drop(columns=["Pclass"], axis=1, inplace=True)
test.drop(columns=["Pclass"], axis=1, inplace=True)
train = pd.concat([train, pd.DataFrame(train_pclass)], axis=1)
test = pd.concat([test, pd.DataFrame(test_pclass)], axis=1)

# rename the columns of the new train df
train = train.rename(columns={0:'1st class',1:'2nd class',2:'3rd class'})
test = test.rename(columns={0:'1st class',1:'2nd class',2:'3rd class'})
#*******************************************************************************************************




















############################################## SEX ATTRIBUTE #######################################
#The 'Sex' attrivute does not have any missing value or nan value
plt.figure(figsize=(10,5))
sns.countplot(x = 'Sex', hue = 'Survived', data = train, palette = 'Paired')
plt.show()

#Let's plot several conditional relationship related the the Pclass attributes
grid = sns.FacetGrid(train, col='Survived', row='Sex', palette='Paired')
grid.map_dataframe(sns.histplot, x='Age')
plt.show()


fg = sns.FacetGrid(train, hue="Sex", aspect=3)
fg.map(sns.kdeplot, "Age", shade=True)
plt.legend()
fg.set(xlim=(0, 80))


#----------------------------------------------------- LabelBinarizer ----------------------------------------------------------------------
# Let's convert the numerical attributes 'male' and 'female' into indicator variables using the LabelBinarizer function
encoder = LabelBinarizer()
train_sex = encoder.fit_transform(train["Sex"])
test_sex = encoder.fit_transform(test["Sex"])

#drop the 'Sex' attrivute for both the data and the test set
#concatenate the new variables into our training and test datasets
#Rq: we have only added the column female because this columns gives us all the necessary information we need 
train.drop(columns=["Sex"], axis=1, inplace=True)
test.drop(columns=["Sex"], axis=1, inplace=True)
train = pd.concat([train, pd.DataFrame(train_sex)], axis=1)
test = pd.concat([test, pd.DataFrame(test_sex)], axis=1)

#rename the columns 'female' into 'Sex' #female = 1, male = 0
train = train.rename(columns={0:'female=1/male=0'})
test = test.rename(columns={0:'female=1/male=0'})























##############################################  AGE ATTRIBUTE #######################################
#the 'Age' attribute has 714 non-null values which gives us 177 null/nan values

#plotting histogram of the Age
plt.hist(train['Age'], bins= 10)
plt.xlabel("Age"); plt.ylabel("Frequence")
plt.show()

#breakdown by age and sex


fg = sns.FacetGrid(train_copy, col='Sex', row='Pclass', hue='Pclass', height=2.5, aspect=2.5)
fg.map(sns.kdeplot, 'Age', shade=True)
fg.map(sns.rugplot, 'Age')
sns.despine(left=True)
fg.set(xlim=(0, 80))

#let's get a better view of statistical properties of the 'Age' attrivute in the train dataset
#and fill the 'nan' values for 'Age' with the mean
for data in datas:
    print(round(data['Age'].describe(),3))
    print(30*'-')

#Let's compute the missing values by replacing them by the median.
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)













##############################################  FARE ATTRIBUTE #######################################
#there is 1 missing value in the test dataset for the 'Fare' attribute.


#plotting histogram of the Fare price pa
f, axes = plt.subplots(1,1, figsize=(10,5))
g1 = sns.histplot(x=train.Fare, color='red', ax= axes)
plt.title("Fare price paid by passengers")
plt.xlabel("Fare"); plt.ylabel("Frequence")
plt.show()

fg = sns.FacetGrid(train, hue="Fare", aspect=3)
fg.map(sns.kdeplot, "Age", shade=True)
plt.legend()
fg.set(xlim=(0, 80))

fg = sns.FacetGrid(train_copy, hue="Sex", aspect=3)
fg.map(sns.kdeplot, "Fare", shade=True)
plt.legend()
fg.set(xlim=(-100, 100))


fg = sns.FacetGrid(train_copy, col='Survived', height=2.5, aspect=2.5)
fg.map(sns.kdeplot, 'Fare', shade=True)
fg.map(sns.rugplot, 'Fare')
sns.despine(left=True)
fg.set(xlim=(-100, 100))

'''
#creating of a sublist of Fare price paid by people that died/survived
Fare_Died = []
Fare_Survived = []
for i in range(0,len(train)):
    if train.Survived[i] == 0: #did not survived
        Fare_Died.append(train.Fare[i])
    else:
        Fare_Survived.append(train.Fare[i])

f, axes = plt.subplots(nrows=1, ncols=1, figsize=(20,10))
g1 = sns.histplot(Fare_Died, color="orange",ax = axes)
plt.title("Fare distribution for the people who did not survive")
plt.show()

f, axes = plt.subplots(1,1, figsize = (20, 10))
g1 = sns.histplot(Fare_Survived, color="blue",ax = axes)
plt.title("Fare distribution for the people who survived")
plt.show()
'''

#since there is 1 missing value in the 'Fare' attrovute in the test dataset, we simply fillna by the mean
test["Fare"].fillna(test['Fare'].mean(), inplace = True) 


'''
# here, we want to break it up into discrete bins by breaking down the data into several ranges
# let's breal k up our Fare features into multiple thresolds (the Q1, Q2, Q3)
train['Fare'] = np.digitize(train["Fare"], bins=[7.910400,14.454200,31.000000], right=True)
test['Fare'] = np.digitize(test["Fare"], bins=[7.895800,14.454200, 31.500000], right=True)

#use of the LabelBinarizer function that gives numerical attributes to text values
encoder = LabelBinarizer()
train_fare = encoder.fit_transform(train["Fare"])
test_fare = encoder.fit_transform(test["Fare"])

#use of the classes_ attrivute to see which index is which class
print(encoder.classes_)

train = pd.concat([train, pd.DataFrame(train_fare)], axis=1)
train = train.drop(columns=["Fare"], axis=1)
test = pd.concat([test, pd.DataFrame(test_fare)], axis=1)
test = test.drop(columns=["Fare"], axis=1)

# rename the columns of the new train df
train = train.rename(columns={0:'Fare range 1',1:'Fare range 2',2:'Fare range 3',3:'Fare range 4'})
test = test.rename(columns={0:'Fare range 1',1:'Fare range 2',2:'Fare range 3',3:'Fare range 4'})
'''











##############################################  EMBARKED ATTRIBUTE #######################################
sns.countplot(y='Embarked', hue='Survived', data=train, orient='h',palette='Set3')
plt.show()

 
#from datavisualisation, we can see that Southhampton is the most frequent embarked port
#so, filling the missing values with 'S' (2 missing values in the train dataset)
train['Embarked'].fillna('S', inplace=True)
test['Embarked'].fillna('S', inplace=True)

fg = sns.FacetGrid(train_copy, col='Survived', row='Embarked', height=2.5, aspect=2.5, palette='Set2')
fg.map(sns.kdeplot, 'Fare', shade=True)
fg.map(sns.rugplot, 'Fare')
sns.despine(left=True)
fg.set(xlim=(-100, 100))

grid = sns.FacetGrid(train_copy, col='Survived', row='Embarked', palette='Set2')
grid.map(sns.barplot, 'Sex', 'Fare', ci=None)
grid.add_legend()
plt.show()


#use of the LabelBinarizer function that gives numerical attributes to text values
# here C, S, Q
encoder = LabelBinarizer()
train_embarked = encoder.fit_transform(train["Embarked"]) #print the classes (C = classe 0, Q = Classe 1, S = Classe 2)
test_embarked = encoder.fit_transform(test["Embarked"])

#use of the classes_ attrivute to see which index is which class
print('The different classes in the Pclass attribute are', encoder.classes_) #class 0 = C / class 1 = Q / class 2 = S

#concatenate with the new dummies variables and delete the 'Embarked' attribute
#Rq: the train_embarked output is an array and we have to convert it to a df
train.drop(columns=["Embarked"], axis=1, inplace=True)
test.drop(columns=["Embarked"], axis=1, inplace=True)
train = pd.concat([train, pd.DataFrame(train_embarked)], axis=1)
test = pd.concat([test, pd.DataFrame(test_embarked)], axis=1)

# rename the columns of the new train df
train = train.rename(columns={0:'C',1:'Q',2:'S'})
test = test.rename(columns={0:'C',1:'Q',2:'S'})










##############################################  SIBSP & PARCH ATTRIBUTES #######################################
#When looking at the correlation matrix, we see that the corr betwee SibSp and parch = 41%   
# so we  create a new attribute 'alone'

def accompanied(x):
        if (x['SibSp'] + x['Parch']) > 0:
            return 1
        else:
            return 0

##for data in datas:
#    data['family'] = data.apply(fam, axis=1)
train['Accompanied'] = train.apply(accompanied, axis = 1)
test['Accompanied'] = test.apply(accompanied, axis = 1)

#drop both the columns Parch and Sibsp
train.drop(columns=['Parch','SibSp'], axis=1, inplace=True)
test.drop(columns=['Parch','SibSp'], axis=1, inplace=True)

#let's plot the histogram of the new attribute vs Survived/Not survuved
plt.figure(figsize=(10,6))
sns.countplot(x='Accompanied', hue='Survived', data = train, palette="YlOrBr")
plt.show()
#those having accompaned have better chance to survive according to the data















##############################################  Name ATTRIBUTES #######################################
# ************************************                                          ******************************************
# ==================================== CREATING NEW FEATURE TITLE EXTRACTING FROM ==========================================
#                                              EXISTING FEATURE NAME                  
# ************************************                                          ************************************  
#we create a 'Title' feature which contains the title of each passenger and drop the Name column


print(train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).unique().size)
print(test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).unique().size)


train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train.drop(columns=['Name'], axis=1, inplace=True)
test.drop(columns=['Name'], axis=1, inplace=True)


#number of titles in the train set
#for data in datas:
#    print(data.Title.value_counts()) #Mr / Miss / Mrs / Master / Dr 
#    print(30*'-')
    
    
print('All the Titles are: \n', set(train.Title) | set(test.Title))

#Let's replace least occuring (1 time) title in the dataset with rare
least_occuring =  ['Countess', 'Capt', 'Dona', 'Dr', 'Ms', 'Major', 'Mme', 
                   'Jonkheer', 'Col', 'Don', 'Lady', 'Sir', 'Mlle', 'Rev']




train['Title'] = train['Title'].replace(least_occuring, "rare")
test['Title'] = test['Title'].replace(least_occuring, "rare")

#Let's take a look at our Titles 
print(train["Title"].value_counts(), test["Title"].value_counts())


#----------------------------------------------------- LabelBinarizer ----------------------------------------------------------------------
encoder = LabelBinarizer() #class 0 = Master / Class 1 = Miss / class 2 = Mr / Class 3 = Mrs / Class 4 = rare
train_title = encoder.fit_transform(train['Title'])
test_title = encoder.fit_transform(test['Title'])
print('The different classes in the Pclass attribute are', encoder.classes_)

#concatenate with the new dummies variables and delete the 'Title' attribute
train.drop(columns=["Title"], axis=1, inplace=True)
test.drop(columns=["Title"], axis=1, inplace=True)
train = pd.concat([train, pd.DataFrame(train_title)], axis=1)
test = pd.concat([test, pd.DataFrame(test_title)], axis=1)

# rename the columns of the new train df
train = train.rename(columns={0:'Master',1:'Miss',2:'Mr',3:'Mrs',4:'rare'})
test = test.rename(columns={0:'Master',1:'Miss',2:'Mr',3:'Mrs',4:'rare'})












##############################################  Ticket ATTRIBUTES #######################################
#I don't see any accurate relation between this variable and the others, so we'll delete it for the moment
train.drop(columns=['Ticket'], axis=1, inplace=True)
test.drop(columns=['Ticket'], axis=1, inplace=True)




########################################## CHECKING FOR MISSING VALUES ##########################
#no missing values in both the datasets
print(train.info())
print(train.isna().sum())
print(test.isna().sum())






# *************************************                         ******************************************
# ==================================== LOOKING FOR CORRELATIONS ==========================================
# *************************************                         ******************************************

#Pearson Correlation matrix
plt.figure(figsize=(15,10))
corr = train.corr()
sns.heatmap(corr, annot=True)
plt.show()

# look at how much each attrivute correlates with the "survived" instance
print(corr["Survived"].sort_values(ascending=False))
# there is a positive linear correlation between the number of person survived and the Sex (female in this case)
# there is a negative linear correlation between the number of person that survived and the Class



# ************************************  FEATURE SCALING WITH STANDARDIZATION                                   ******************************************
# ====================================  ==========================================
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

train.drop(['Survived'], axis=1, inplace=True)
X_train = standard_scaler.fit_transform(train)
y_train = survived_train
X_test = standard_scaler.fit_transform(test)



###############################################################################################
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

classifiers = ['LogisticRegression','SGDClassifier','DecisionTreeClassifier',
               'RandomForestClassifier', 'AdaBoostClassifier', 'KNeighborsClassifier']



models = [LogisticRegression(), SGDClassifier(random_state=42), DecisionTreeClassifier(random_state=42),
          RandomForestClassifier(), AdaBoostClassifier(DecisionTreeClassifier(random_state=42)), KNeighborsClassifier()]
            
                                         
#voting_clf = VotingClassifier(estimators=[('lr', LogisticRegression()), ('sgd', SGDClassifier(random_state=42)), 
#                                                                 ('dtc', DecisionTreeClassifier(random_state=42)),
#                                                                 ('rfc', RandomForestClassifier())], voting='hard')

#from sklearn.metrics import accuracy_score

#for clf in (LogisticRegression(), SGDClassifier(random_state=42), DecisionTreeClassifier(random_state=42), RandomForestClassifier(), voting_clf):
#    clf.fit(X_train, y_train)
#    y_pred = clf.predict(X_train)
#    print(clf.__class__.__name__, accuracy_score(y_train, y_pred))
                                                     
                                                     
                                                                                                         
#Let's use cross_val_score() method to evaluate our model using K-fold cross-validation
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
################################################################################################




##########################################################################################
#Use of cross_validate function that allows us to use multiple scores / It is best using this instead of what is up
j =0
results = {}
cross_validate_ = {}
for i in models:
    model = i
    scoring = ['accuracy','precision','f1']
    cross_validate_[classifiers[j]] = cross_validate(model, X_train, y_train, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=1), scoring=scoring, n_jobs=-1)
    results[j] = pd.DataFrame(cross_validate_[classifiers[j]])
    j += 1


j =0
#test_precision_LG, test_precision_SGD, test_precision_DTC, test_precision_RFC, test_precision_ADA, test_precision_KNN = []
for i in models:
    test_precision_ = {}
    model = i
    for index in range(0,6):
        test_precision_[classifiers[j]] = pd.DataFrame(results[0]['test_precision']).T
    j += 1



test_precision_LG = pd.DataFrame(results[0]['test_precision']).T
test_precision_SGD = pd.DataFrame(results[1]['test_precision']).T
test_precision_DTC = pd.DataFrame(results[2]['test_precision']).T
test_precision_RFC = pd.DataFrame(results[3]['test_precision']).T
test_precision_ADA = pd.DataFrame(results[4]['test_precision']).T
test_precision_KNN = pd.DataFrame(results[5]['test_precision']).T
test_precision_all = pd.concat([test_precision_LG, test_precision_SGD, test_precision_DTC, test_precision_RFC, test_precision_ADA, test_precision_KNN], axis=0)
test_precision_all.index = classifiers
test_precision_mean = test_precision_all.mean(axis=1)
test_precision_all= pd.concat([test_precision_all, test_precision_mean], axis=1)
test_precision_all.columns = ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5', 'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9', 'Fold 10','Mean']

test_accuracy_LG = pd.DataFrame(results[0]['test_accuracy']).T
test_accuracy_SGD = pd.DataFrame(results[1]['test_accuracy']).T
test_accuracy_DTC = pd.DataFrame(results[2]['test_accuracy']).T
test_accuracy_RFC = pd.DataFrame(results[3]['test_accuracy']).T
test_accuracy_ADA = pd.DataFrame(results[4]['test_accuracy']).T
test_accuracy_KNN = pd.DataFrame(results[5]['test_accuracy']).T
test_accuracy_all = pd.concat([test_accuracy_LG, test_accuracy_SGD, test_accuracy_DTC, test_accuracy_RFC, test_accuracy_ADA, test_accuracy_KNN], axis=0)
test_accuracy_all.index = classifiers
test_accuracy_mean = test_accuracy_all.mean(axis=1)
test_accuracy_all= pd.concat([test_accuracy_all, test_accuracy_mean], axis=1)
test_accuracy_all.columns = ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5', 'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9', 'Fold 10','Mean']

##########################################################################################


##########################################################################################
# A much better way to evaluate the performance is to look at the confusion matrix
from sklearn.model_selection import cross_val_predict #performs K folds cross validation but instead of returning the evaluation scores, it returns the predictions made on each test fold
from sklearn.metrics import confusion_matrix, roc_auc_score

j =0
cross_val_predictions = {}
conf_matrix = {}
roc_auc_scores = {}
for i in models:
    model = i
    cross_val_predictions[classifiers[j]] = cross_val_predict(model, X_train, y_train, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=1))
    conf_matrix[classifiers[j]] = confusion_matrix(y_train, cross_val_predictions[classifiers[j]])
    plt.show()
    j += 1
    

    
##########################################################################################



# ************************************  FINE TUNE YOUR MODEL ******************************************
# ====================================  Grid Search ==========================================
#let search for the best possible combinations of hyperparameters values for RandomFirestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np             

param_grid = {'n_estimators': [int(x) for x in np.linspace(200, 2000, num = 10)], 'max_features': ['auto', 'sqrt'], 
              'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)],
              'criterion' :['gini', 'entropy']}


grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=1))

best_model = grid_search.fit(X_train, y_train)
print(best_model.best_params_)         

classifier_best = RandomForestClassifier(random_state=42, n_estimators=300, criterion='entropy',max_depth=6, max_features='sqrt')
classifier_best.fit(X_train, y_train)

X_test_pred = classifier_best.predict(X_test)
final_results = pd.DataFrame({'PassengerId': PassengerId_test, 'Survived': X_test_pred})
final_results



