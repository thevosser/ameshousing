import pandas as pd
import os
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

os.chdir("/Users/ericvoss/Desktop/house-prices-advanced-regression-techniques/")

# Load dataframes
traindf = pd.read_csv("train.csv")
testdf = pd.read_csv("test.csv")
colvalsdf = pd.read_excel("data_description_table.xlsx")

# Reduce clean values for ordinals - find better method than manual
colvalsdf = colvalsdf.dropna(subset=['AssignedValue'])

# Pivot to then convert to dict to then change values on the other dfs
colvalpvt = colvalsdf.pivot(index="ColValue", columns="Column", values="AssignedValue")

# Make dict from pivto
colvaldict = colvalpvt.to_dict("dict")

# Use replace to clean the values
trainclndf = traindf.replace(colvaldict)
testclndf = testdf.replace(colvaldict)

# Copy cleaned data frame
ohedf = trainclndf.copy()
ohedftest = testclndf.copy()

# Loop through all columns - very poorly written
for column in trainclndf.columns:
    print(column)
    if column in list(colvaldict.keys()):
        print(column)
        ohedf.drop(columns=(column), axis=0, inplace=True)
    else:
        for row in range(1, 50): #test first 9 rows if text is present
            if type(ohedf[column][row]) != str and ohedf[column][row] is not np.nan:
                print(type(ohedf[column][row]))
                ohedf.drop(columns=(column), axis=0, inplace=True)
                break


# One Hot Encode remaining text columns
ohedfcln = pd.get_dummies(ohedf)
ohedfclntest = pd.get_dummies(ohedftest[ohedf.columns])

for col in ohedfcln.columns:
    if col not in ohedfclntest.columns:
        ohedfclntest[col] = 0

# Drop all the columns that were encoded from the other df
trainclndf.drop(columns=(ohedf.columns), axis=0, inplace=True)
testclndf.drop(columns=(ohedf.columns), axis=0, inplace=True)

# merge the two dfs
maintraindf = trainclndf.join(ohedfcln)
maintestdf = testclndf.join(ohedfclntest)

corrmtx = maintraindf.corr()

# correlation matrix
import matplotlib.pyplot as plt

plt.matshow(maintraindf.corr())
plt.show()

# split to analysis best columns
X = maintraindf.loc[:, ~maintraindf.columns.isin(["Id","SalePrice"])]  #independent columns
y = maintraindf.loc[:, maintraindf.columns == "SalePrice"]    #target column i.e price range

X = X.fillna(0) #probably should be scaled like below in PCA

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=25)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(25,'Score'))  #print 10 best features

corrmtx2 = maintraindf[list(featureScores.nlargest(25,'Score')["Specs"])].corr()


########
# Try PCA
# https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)

from sklearn.decomposition import PCA
pca_30 = PCA(n_components=45, random_state=2020)
pca_30.fit(Xscaled)
X_pca_30 = pca_30.transform(X)
pca_30.explained_variance_ratio_ * 100
plt.plot(np.cumsum(pca_30.explained_variance_ratio_ * 100))
plt.show()


########
# Try Recursive Feature Elimination RFE
# Feature Extraction with RFE
# from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load data
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = read_csv(url, names=names)
# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
# feature extraction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, n_features_to_select= 3, step=1)
fit = rfe.fit(X, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


###### need to score how well it fits to training set
# fit.

#predict test set
# predX = testdf[fit.get_feature_names_out()].fillna(0)
maintestdf.drop(columns="Id", inplace=True)
fit.predict(maintestdf.fillna(0))

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=10)
treefit = model.fit(X, y)
print(treefit.feature_importances_)





