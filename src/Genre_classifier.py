import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
#TODO maybe try on more data - accuracy is so bad, classes aren't balanced
model = SentenceTransformer("all-MiniLM-L6-v2")

DIR = '/Users/bahar/Documents/Projects/Books_geek/resources'
df_books = pd.read_parquet(os.path.join(DIR, 'goodreads_cleaned_7000.parquet'))
print(f'shape:{df_books.shape}')
'''7000: shape:(4354, 30)
'''

text_features = ['title', 'authors', 'description']
num_features = ['publication_year']

#---Split the data set---
X = df_books[['title','description',
              'publication_year']]
y = df_books['genre']

print("Samples per genre: ")
print(y.value_counts())


# Encode target
le = LabelEncoder()
y_final = le.fit_transform(y.values)
#each genre with their label
for i, genre in enumerate(le.classes_):
    print(f"{i}: {genre}")



X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=1000, random_state=42) #TODO split in a way the genre in test is balanced

scaler = StandardScaler()
num_train = scaler.fit_transform(X_train[['publication_year']])
num_test = scaler.transform(X_test[['publication_year']])

# embeddings - just encode each split separately
desc_train = model.encode(X_train['description'].tolist())
desc_test = model.encode(X_test['description'].tolist())

title_train = model.encode(X_train['title'].tolist())
title_test = model.encode(X_test['title'].tolist())



# combine numerical + text embeddings into final feature matrix
X_train_final = np.hstack([num_train, desc_train, title_train])
X_test_final = np.hstack([num_test, desc_test, title_test])

#---Decision tree classifier---

dtree = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree.fit(X_train_final, y_train)
y_pred = dtree.predict(X_test_final)

# convert numeric predictions back to genre names
#y_pred_labels = le.inverse_transform(y_pred)
#print(y_pred_labels)

#Evaluation metrics: accuracy and F1 score
dtree_acc = accuracy_score(y_test, y_pred)
dtree_cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average=None) # f1_score = 2 * (Precision * Recall)/Precision + Recall
print("report:" + classification_report(y_test, y_pred, target_names=le.classes_))


print("sklearn depth:", dtree.get_depth())
print("sklearn leaves:", dtree.get_n_leaves())
print("unique sklearn preds:")
print(np.unique(y_pred))
print("Decision Tree Accuracy:", dtree_acc)
print("F1 score per class: ", f1_score(y_test, y_pred, average=None))

plt.figure(figsize=(4, 3))
sns.heatmap(dtree_cm, annot=True, cmap="Blues", fmt="d")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

