from datasets import load_dataset, DatasetDict
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# Veri setini yükleyin
ds = load_dataset("sepidmnorozy/Turkish_sentiment")

# Veriyi eğitim, test ve doğrulama setlerine ayırma
ds = ds['train'].train_test_split(test_size=0.2, seed=42)
ds_valid_test = ds['test'].train_test_split(test_size=0.5, seed=42)

# final_ds'i oluşturun
final_ds = DatasetDict({
    'train': ds['train'],
    'validation': ds_valid_test['train'],
    'test': ds_valid_test['test']
})

# Eğitim ve doğrulama verilerini ayırın
X_train = final_ds['train']['text']
y_train = final_ds['train']['label']
X_val = final_ds['validation']['text']
y_val = final_ds['validation']['label']

# Metin verisini token'lara ayırma (kelimeleri ayırma)
def preprocess_text(text):
    return text.lower().split()

# Veriyi işleyip token'lara ayırma
X_train_tokens = [preprocess_text(text) for text in X_train]
X_val_tokens = [preprocess_text(text) for text in X_val]

# Word2Vec Modeli
model = Word2Vec(sentences=X_train_tokens, vector_size=300, window=15, min_count=10, workers=4)

# Veriyi vektörlere dönüştürme
def vectorize_text(tokens, model):
    vector = np.zeros(300)  # 100 boyutlu bir vektör
    valid_words = 0
    for word in tokens:
        if word in model.wv:
            vector += model.wv[word]  # Kelimenin embedding'ini al
            valid_words += 1
    if valid_words > 0:
        vector /= valid_words  # Vektörü normalize et (ortalama al)
    return vector

# Eğitim verisini vektör haline getir
X_train_vectors = np.array([vectorize_text(text, model) for text in X_train_tokens])
X_val_vectors = np.array([vectorize_text(text, model) for text in X_val_tokens])

# Lojistik Regresyon Modeli
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vectors, y_train)

# Tahmin yap
y_pred = clf.predict(X_val_vectors)

# Sonuçları yazdır
print(classification_report(y_val, y_pred))
