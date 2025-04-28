import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import joblib
from data_cleaner import clean_dataset

# Carregar dados
df = pd.read_csv("data/amazon-reviews/train.csv", header=None, names=["Label", "Title", "Text"])

# Pré-processamento
df['Sentiment'] = df['Label'].map({1: 0, 2: 1})
df['Full_Text'] = df['Title'].astype(str) + " " + df['Text'].astype(str)
df = clean_dataset(df, 'Full_Text')

# Separar dados
X_train, X_test, y_train, y_test = train_test_split(df['Full_Text'], df['Sentiment'], test_size=0.2, random_state=42)

# Vetorização
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Treinamento
model = LogisticRegression(solver='saga', max_iter=2000)
model.fit(X_train_vec, y_train)

# Avaliação
y_pred = model.predict(X_test_vec)
print("Acurácia:", accuracy_score(y_test, y_pred))

# Salvar o modelo e o vetor
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Exemplo de predição
def predict_sentiment(text):
    cleaned = TextPreprocessor().clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)
    return "Positiva" if prediction[0] == 1 else "Negativa"

# Testar a predição
print(predict_sentiment("This product is amazing, I loved it!"))
print(predict_sentiment("Terrible quality, I'm disappointed."))

# Matrizes de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativa', 'Positiva'], yticklabels=['Negativa', 'Positiva'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Curva ROC
y_prob = model.predict_proba(X_test_vec)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Métricas de Avaliação
report = classification_report(y_test, y_pred)
print(report)
