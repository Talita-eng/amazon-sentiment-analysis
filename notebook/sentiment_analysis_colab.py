!pip install kagglehub joblib scikit-learn

import kagglehub

path = kagglehub.dataset_download("kritanjalijain/amazon-reviews")
print("Path:", path)

%%writefile data_cleaner.py
import re
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

def clean_dataset(df, text_column):
    cleaner = TextPreprocessor()
    df[text_column] = df[text_column].astype(str).apply(cleaner.clean_text)
    return df


import os
os.listdir("/root/.cache/kagglehub/datasets/kritanjalijain/amazon-reviews/versions/2")

import pandas as pd
from data_cleaner import clean_dataset

df = pd.read_csv("/root/.cache/kagglehub/datasets/kritanjalijain/amazon-reviews/versions/2/train.csv", header=None, names=["Label", "Title", "Text"])

# Ajustar os rótulos: 1 (negativo) → 0, 2 (positivo) → 1
df['Sentiment'] = df['Label'].map({1: 0, 2: 1})

# Usar apenas a coluna 'Text' e 'Title' para análise

df['Full_Text'] = df['Title'].astype(str) + " " + df['Text'].astype(str)
df = clean_dataset(df, 'Full_Text')


# Exibir as primeiras linhas
df.head()

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# BALANCEAMENTO DE CLASSES
# ==========================================

# Contagem das classes
class_counts = df['Sentiment'].value_counts()

# Gráfico de barras
plt.figure(figsize=(6,5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='Blues')
plt.xlabel('Sentimento')
plt.ylabel('Número de Avaliações')
plt.title('Distribuição de Avaliações (Negativa vs Positiva)')
plt.xticks([0, 1], ['Negativa', 'Positiva'])
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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


import joblib

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

loaded_model = joblib.load("sentiment_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_sentiment(text):
    from data_cleaner import TextPreprocessor
    cleaner = TextPreprocessor()
    cleaned = cleaner.clean_text(text)
    vec = loaded_vectorizer.transform([cleaned])
    prediction = loaded_model.predict(vec)
    return "Positiva" if prediction[0] == 1 else "Negativa"

# Exemplo
print(predict_sentiment("This product is amazing, I loved it!"))
print(predict_sentiment("Terrible quality, I'm disappointed."))

from sklearn.metrics import confusion_matrix, roc_curve, auc



# ==========================================
# MATRIZ DE CONFUSÃO
# ==========================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativa', 'Positiva'], yticklabels=['Negativa', 'Positiva'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# ==========================================
# CURVA ROC E AUC
# ==========================================
y_prob = model.predict_proba(X_test_vec)[:,1]  # Probabilidades da classe positiva
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

from sklearn.metrics import classification_report

# Calcular o relatório com as métricas de precisão, recall e F1-score
report = classification_report(y_test, y_pred)
print(report)

# Função para registrar as métricas periodicamente (exemplo)
def log_metrics(y_true, y_pred, file_path='metrics_log.csv'):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df['timestamp'] = pd.to_datetime('now')

    # Salvar ou atualizar o arquivo de log
    try:
        existing_data = pd.read_csv(file_path)
        updated_data = pd.concat([existing_data, df], ignore_index=True)
    except FileNotFoundError:
        updated_data = df

    updated_data.to_csv(file_path, index=False)

# Exemplo de como usá-la após realizar uma previsão
log_metrics(y_test, y_pred)

from google.colab import files
files.download('sentiment_model.pkl')

files.download('tfidf_vectorizer.pkl')