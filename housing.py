import pandas as pd
import os
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# os.chdir("/Users/ericvoss/Desktop/house-prices-advanced-regression-techniques/")
os.chdir("C:\\Users\\PL1Z429\\Box\\Personal Workspace -- Eric Voss\\DataScience\\HousingPrices")

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
    if column in list(colvaldict.keys()):
        ohedf.drop(columns=(column), axis=0, inplace=True)
    else:
        for row in range(1, 50): #test first 9 rows if text is present
            if type(ohedf[column][row]) != str and ohedf[column][row] is not np.nan:
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


# split the df for model fitting
X = maintraindf.loc[:, ~maintraindf.columns.isin(["Id","SalePrice"])]  #independent columns
y = maintraindf.loc[:, maintraindf.columns == "SalePrice"]    #target column i.e price range

X = X.fillna(0) #probably should be scaled like below in PCA

######
#####try scaling before selecting selectKbest
######



#apply SelectKBest class to extract top 10 best features
nFeats = 10 #number of features to limit to
bestfeatures = SelectKBest(score_func=chi2, k=nFeats)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(nFeats,'Score'))  #print 10 best features

# create a correlation matrix
corrmtx2 = maintraindf[list(featureScores.nlargest(nFeats,'Score')["Specs"])].corr()


######
#####Limit features to SelectKBest selction
#####

list(featureScores.nlargest(10,'Score')["Specs"])

#########
######train test split here
#########

#predict test set
# predX = testdf[fit.get_feature_names_out()].fillna(0)
# maintestdf.drop(columns="Id", inplace=True)

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=10)
treefit = model.fit(X, y)
print(treefit.feature_importances_)

######
#####Score model
######





# add predictions to test
testprices = pd.DataFrame(treefit.predict(maintestdf.fillna(0)))
testoutput = pd.concat([testprices, maintestdf], axis=1, join="inner")

testoutput.rename({0:"SalePrice"}, axis=1)

# output to file
testoutput.to_csv("testpred.csv")










########
# Try PCA
# https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)

from sklearn.decomposition import PCA
pca_30 = PCA(n_components=150, random_state=2020)
pca_30.fit(Xscaled)
X_pca_30 = pca_30.transform(X)
pca_30.explained_variance_ratio_ * 100
plt.plot(np.cumsum(pca_30.explained_variance_ratio_ * 100))
plt.show()





