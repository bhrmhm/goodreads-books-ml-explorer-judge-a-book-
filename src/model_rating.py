import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.decomposition import PCA


from src.regression_models.Linear_regression import MultipleLinearRegression
from src.regression_models.Regression_tree import RegressionTree

model = SentenceTransformer("all-MiniLM-L6-v2")

DIR = '/Users/bahar/Documents/Projects/Books_geek/resources'
df_books = pd.read_parquet(os.path.join(DIR, 'goodreads_cleaned_2000.parquet'))

text_features = ['title', 'authors', 'genre', 'description', 'language_code']
num_features = ['ratings_count', 'text_reviews_count','num_pages', 'publication_year']

#---Split the data set---
X = df_books[['description', 'authors',
              'ratings_count', 'text_reviews_count', 'num_pages',
              'publication_year']]
y = df_books['average_rating']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#---encode text feature and scale numerical features---
# scale numerical features - fit ONLY on train
scaler = StandardScaler()
num_train = scaler.fit_transform(X_train[num_features])
num_test = scaler.transform(X_test[num_features])

# embeddings - just encode each split separately
desc_train = model.encode(X_train['description'].tolist())
desc_test = model.encode(X_test['description'].tolist())

#title_train = model.encode(X_train['title'].tolist())
#title_val = model.encode(X_val['title'].tolist())
#title_test = model.encode(X_test['title'].tolist())


# combine numerical + text embeddings into final feature matrix
X_train_final = np.hstack([num_train, desc_train])
X_test_final = np.hstack([num_test, desc_test])


print("y_train mean:", y_train.mean())
print("y_test mean:", y_test.mean())
print("y_train std:", y_train.std())
print("y_test std:", y_test.std())
print("y_train variance:", np.var(y_train))
print("X_train_final shape:", X_train_final.shape)
#PCA - reduce dimension
pca = PCA(n_components=50)
X_train_final_pca = pca.fit_transform(X_train_final)
X_test_final_pca = pca.transform(X_test_final)
print("explained variance:", sum(pca.explained_variance_ratio_))
print("X_train_final shape:", X_train_final_pca.shape)


#---Linear regression model---
'''model = LinearRegression()
model.fit(X_train_final, y_train)
y_pred = model.predict(X_test_final_pca)
print("Model 1 pred: ", y_pred)
print("RMSE:", np.sqrt(np.mean((y_test - y_pred) ** 2)))
print("R2:", 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))'''

"""RMSE: 0.9928490668386858 lower the better
R2: -4.231066555060551   higher the better (closer to one)"""
"""RMSE: 0.5294107000052394
R2: -0.49020218678277083"""
#after PCA
"""RMSE: 0.4326145352939926
R2: 0.004910872178388814"""

'''myModel = MultipleLinearRegression()
myModel.fit(X_train_final_pca, y_train)
y_pred = myModel.predict(X_test_final_pca)
print("Model 2 pred: ", y_pred)
print("y_pred range:", y_pred.min(), y_pred.max())
print("W range:", myModel.W.min(), myModel.W.max())
print("b:", myModel.b)
print("RMSE:", myModel.rmse(y_test, y_pred))
print("R2:", myModel.r2score(y_test, y_pred))'''

"""RMSE: 0.429424723822907
R2: 0.019531022148843724"""

#Some results on linear regression:
"""y_pred range: 3.780898398463432 3.984172482131539
y_test mean: 3.8583516483516482
predicting values clustered around the mean -
which is what a weak linear model does when it can't find strong signal"""
#todo: export data about prediction and actual y - soem kind of visualization


#---Regression tree model---

regr_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
regr_tree.fit(X_train_final_pca, y_train)

y_pred = regr_tree.predict(X_test_final_pca)
print("sklearn depth:", regr_tree.get_depth())
print("sklearn leaves:", regr_tree.get_n_leaves())
print("Model 1 pred: ", y_pred)
print("y_pred range:", y_pred.min(), y_pred.max())
print("unique sklearn preds:", len(np.unique(y_pred)))
print("RMSE:", np.sqrt(np.mean((y_test - y_pred) ** 2)))
print("R2:", 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))
#max_depth= 3 + PCA
"""y_pred range: 2.89 3.8965186915887844
RMSE: 0.44099282901811754
R2: -0.03400543610833062"""

#max_depth= 3 + PCA
"""y_pred range: 3.8688444924406027 3.9885624999999996
unique sklearn preds: 2
RMSE: 0.4352485480737926
R2: -0.0072433972649526"""


my_regr_tree = RegressionTree(max_depth=3)
my_regr_tree.fit(X_train_final_pca, y_train)

y_pred = my_regr_tree.predict(X_test_final_pca)
print("Model 2 pred: ", y_pred)
print("y_pred range:", y_pred.min(), y_pred.max())
print("unique my preds:", len(np.unique(y_pred)))
print("RMSE:", my_regr_tree.rmse(y_test, y_pred))
print("R2:", my_regr_tree.r2_score(y_test, y_pred))
#max_depth= 5
"""y_pred range: 1.69 3.8965186915887853
RMSE: 0.4653686708277204
R2: -0.1514737781939106"""

#max_depth= 3 + PCA
"""y_pred range: 3.868844492440605 3.9885625000000005
unique my preds: 2
RMSE: 0.4352485480737927
R2: -0.007243397264953266
"""
#finding the best max_depth
'''depths = [3, 5, 10, 15, 20]
for depth in depths:
    dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
    dt.fit(X_train_final_pca, y_train)
    y_pred = dt.predict(X_test_final_pca)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    print(f"depth={depth} RMSE={rmse:.4f} R2={r2:.4f}")'''

"""
y_train variance: 0.14668828760205369
depth=3 RMSE=0.4674 R2=-0.1614
depth=5 RMSE=0.4792 R2=-0.2208

R2 gets more negative as depth increases, which is a classic sign of overfitting
Variance of 0.14 — ratings are clustered very tightly around the mean, very little signal to split on
"""
"""with PCA:
depth=3 RMSE=0.4365 R2=-0.0130
depth=5 RMSE=0.4495 R2=-0.0742
depth=10 RMSE=0.4867 R2=-0.2592
depth=15 RMSE=0.5376 R2=-0.5366
depth=20 RMSE=0.5623 R2=-0.6808
a little bit better"""