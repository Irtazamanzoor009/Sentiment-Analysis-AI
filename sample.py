from sklearn.svm import LinearSVC
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import SGDClassifier
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# import scikitplot as skplt
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("data.csv")

dataset.head()

dataset.shape

dataset.isnull().sum()

dataset = dataset[['brand','manufacturer','reviews.didPurchase','reviews.rating', 'reviews.text']]

dataset.isnull().sum()

dataset['reviews.didPurchase'] = dataset['reviews.didPurchase'].fillna('Not Avialable')

dataset = dataset.dropna()

data = dataset['reviews.rating'].value_counts()

sns.barplot(x=data.index, y=data.values)

ax_plt = sns.countplot(dataset['reviews.didPurchase'])
ax_plt.set_xlabel(xlabel='No. of Reviews', fontsize=12)
ax_plt.set_ylabel(ylabel="User's Reviews", fontsize=12)
ax_plt.set_title('Accurate No. of Reviews', fontsize=12)
ax_plt.tick_params(labelsize=11)

stopwords = set(STOPWORDS)
def wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=250,
        max_font_size=30,
        scale=2,
        random_state=5
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)
    plt.imshow(wordcloud)
    plt.show()

wordcloud(dataset['reviews.text'])

data=dataset['reviews.text']
train_data=dataset['reviews.text']
y_target=dataset['reviews.rating'].map({1:'Unhappy',2:'Unhappy',3:'Ok',4:'Happy',5:'Happy'})
vectorize_word = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',stop_words='english',ngram_range=(1, 1),max_features=10000)
vectorize_word.fit(data)
train_features_word = vectorize_word.transform(train_data)
vectorize_char = TfidfVectorizer(sublinear_tf=True,strip_accents='unicode',analyzer='char',stop_words='english',ngram_range=(2, 6),max_features=50000)
vectorize_char.fit(data)
train_features_char = vectorize_char.transform(train_data)
train_features = hstack([train_features_char, train_features_word])
X_train, X_test, y_train, y_test = train_test_split(train_features, y_target,test_size=0.3,random_state=101,shuffle=True)

lsvm = LinearSVC(class_weight='balanced')
l = lsvm.fit(X_train,y_train)
pred_train = l.predict(X_train)
print("Accuracy Train: {}".format(accuracy_score(y_train,pred_train)))
print(classification_report(y_train,pred_train))

pred_test=l.predict(X_test)
print("Accuracy Test : {}".format(accuracy_score(y_test,pred_test)))
print(classification_report(y_test,pred_test))

cm = confusion_matrix(y_test, pred_test, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Happy', 'Ok', 'Unhappy'], yticklabels=['Happy', 'Ok','Unhappy'])
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

svm1=SGDClassifier(class_weight='balanced',n_jobs=-1, max_iter=300)
svm1.fit(X_train,y_train)
pred_train_sgd=svm1.predict(X_train)
print("Accuracy Train: {}".format(accuracy_score(y_train,pred_train_sgd)))
print(classification_report(y_train,pred_train_sgd))
pred_test_sgd=svm1.predict(X_test)
print("Accuracy Test: {}".format(accuracy_score(y_test,pred_test_sgd)))
print(classification_report(y_test,pred_test_sgd))
cm = confusion_matrix(y_test, pred_test_sgd, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Happy', 'Ok', 'Unhappy'], yticklabels=['Happy', 'Ok','Unhappy'])
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

# Dictionary to store models and their accuracy scores
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LinearSVC": LinearSVC(class_weight='balanced'),
    "SGDClassifier": SGDClassifier(class_weight='balanced', n_jobs=-1, max_iter=300),
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=300),
    "RandomForest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
    "NaiveBayes": MultinomialNB(),
}

# Dataframe to store results
results = {"Model": [], "Train Accuracy": [], "Test Accuracy": []}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    # Accuracy scores
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    
    # Classification report
    print(f"{model_name} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, pred_test))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, pred_test, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=['Happy', 'Ok', 'Unhappy'], yticklabels=['Happy', 'Ok', 'Unhappy'])
    plt.title(f"{model_name} - Normalized Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()
    
    # Save results
    results["Model"].append(model_name)
    results["Train Accuracy"].append(train_acc)
    results["Test Accuracy"].append(test_acc)

# Plot comparison of model accuracy
results_df = pd.DataFrame(results)
plt.figure(figsize=(12, 6))
sns.barplot(x="Model", y="Test Accuracy", data=results_df, palette="viridis")
plt.title("Comparison of Test Accuracy for Different Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.show()


