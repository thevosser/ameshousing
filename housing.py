import pandas as pd
import os
import numpy as np
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import statsmodels.api as sm

# os.chdir("/Users/ericvoss/Desktop/house-prices-advanced-regression-techniques/")
# os.chdir("C:\\Users\\PL1Z429\\Box\\Personal Workspace -- Eric Voss\\DataScience\\HousingPrices")
os.chdir(r'C:\Users\vosser\iCloudDrive\DSLearning\Kaggle-Ameshousing'.replace('\\', '/'))

# Load dataframes
traindf = pd.read_csv("train.csv")
testdf = pd.read_csv("test.csv")
colvalsdf = pd.read_excel("data_description_table.xlsx")

# Reduce clean values for ordinals - find better method than manual
colvalsdf = colvalsdf.dropna(subset=['AssignedValue'])

# Pivot to then convert to dict to then change values on the other dfs
colvalpvt = colvalsdf.pivot(index="ColValue", columns="Column", values="AssignedValue")

# Make dict from pivot
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

##################
# Use f_regression to get p_values for features and remove anything > 0.05 for easy feature selection
pvalues = f_regression(X,y)[1]

colcleaner = pd.DataFrame(pvalues)
colcleaner.rename({0: "p-value"}, axis=1, inplace=True)
colcleaner["colnames"] = X.columns
colcleaner = colcleaner[colcleaner["p-value"] > 0.05]

# Drop all the columns that were > 0.05
X.drop(columns=(colcleaner['colnames'].values), inplace=True)


# Run a SKLearn Linear Regression
reg = LinearRegression()
results = reg.fit(X,y)

# Get r2 and adjusted r2 from sklearn
r2 = reg.score(X, y)
adjr2 = 1 - (1 - r2) * ( (X.shape[0] - 1) / (X.shape[0] - (X.shape[1] - 1) - 1 ) )

# Run a statsmodel linear regression to look at the output table
x1 = sm.add_constant(X)
results = sm.OLS(y, x1).fit()
results.summary()
#stats model shows a lot of p-values higher than 0.05 might need to go back and check the work above

# Output values on training set to compare
traindf["predicted_price"] = reg.predict(X)

traindf["perc_off"] = traindf["predicted_price"] / traindf["SalePrice"] 

traindf["perc_off"].mean()

traindf.to_excel("trainpredicted.xlsx")

#Still not great, but a pretty good model, the predictions have some outliers, need to learn more what is doing that.

# check MAE - have not learned this, but wanted to see
from sklearn.metrics import mean_absolute_error
score = mean_absolute_error(traindf["SalePrice"], traindf["predicted_price"])
print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))

# check RSME - have not learned this, but wanted to see
from sklearn.metrics import mean_squared_error
score = np.sqrt(mean_absolute_error(traindf["SalePrice"], traindf["predicted_price"]))
print("The Mean Absolute Error of our Model is {}".format(round(score, 2)))









# ######
# #####try scaling before selecting selectKbest
# ######



# #apply SelectKBest class to extract top 10 best features
# nFeats = 10 #number of features to limit to
# bestfeatures = SelectKBest(score_func=chi2, k=nFeats)
# fit = bestfeatures.fit(X,y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(nFeats,'Score'))  #print 10 best features

# # create a correlation matrix
# corrmtx2 = maintraindf[list(featureScores.nlargest(nFeats,'Score')["Specs"])].corr()


# ######
# #####Limit features to SelectKBest selction
# #####

# list(featureScores.nlargest(10,'Score')["Specs"])

# #########
# ######train test split here
# #########

# #predict test set
# # predX = testdf[fit.get_feature_names_out()].fillna(0)
# # maintestdf.drop(columns="Id", inplace=True)

# from sklearn.ensemble import ExtraTreesClassifier
# model = ExtraTreesClassifier(n_estimators=10)
# treefit = model.fit(X, y)
# print(treefit.feature_importances_)

# ######
# #####Score model
# ######





# # add predictions to test
# testprices = pd.DataFrame(treefit.predict(maintestdf.fillna(0)))
# testoutput = pd.concat([testprices, maintestdf], axis=1, join="inner")

# testoutput.rename({0:"SalePrice"}, axis=1)

# # output to file
# testoutput.to_csv("testpred.csv")










# ########
# # Try PCA
# # https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X)
# Xscaled = scaler.transform(X)

# from sklearn.decomposition import PCA
# pca_30 = PCA(n_components=150, random_state=2020)
# pca_30.fit(Xscaled)
# X_pca_30 = pca_30.transform(X)
# pca_30.explained_variance_ratio_ * 100
# plt.plot(np.cumsum(pca_30.explained_variance_ratio_ * 100))
# plt.show()




