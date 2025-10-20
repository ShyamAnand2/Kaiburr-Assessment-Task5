import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download NLTK resources (one-time)
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load dataset
df = pd.read_csv('consumer_complaints.csv', low_memory=False)

# Only keep rows where both text and the class label exist
df = df[['consumer_complaint_narrative', 'product']].dropna().sample(5000, random_state=42)

# Set proper column names for generic usage
df.columns = ['Consumer complaint narrative', 'Product']

# For reduced clutter, pick only the largest ~8 categories (optional, makes confusion matrix clearer)
top_classes = df['Product'].value_counts().nlargest(8).index.tolist()
df = df[df['Product'].isin(top_classes)]

# Encode target
df['Product_Code'] = df['Product'].astype('category').cat.codes
classes = df['Product'].astype('category').cat.categories

# Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english') and len(word) > 2]
    return " ".join(tokens)

df['Cleaned'] = df['Consumer complaint narrative'].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['Cleaned'], df['Product_Code'], test_size=0.2, stratify=df['Product_Code'], random_state=42
)

# TF-IDF feature extraction
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training and evaluation
results = {}
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear')
}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc*100:.2f}%")
    results[name] = acc
    print(classification_report(y_test, y_pred, target_names=classes))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

best_model_name = max(results, key=results.get)
print(f"Best performing model: {best_model_name} with accuracy {results[best_model_name]*100:.2f}%")

# Sample prediction
sample_texts = [
    "The credit card company charged me an unfair late fee.",
    "My mortgage servicer is refusing proper documentation.",
    "Debt collector keeps calling my workplace."
]
sample_clean = [clean_text(t) for t in sample_texts]
sample_vec = vectorizer.transform(sample_clean)
model = models[best_model_name]
preds = model.predict(sample_vec)
for text, pred in zip(sample_texts, preds):
    print(f"Complaint: '{text}' => Predicted Category: {classes[pred]}")
