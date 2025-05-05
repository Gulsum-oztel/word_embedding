from datasets import DatasetDict
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

ds = load_dataset("sepidmnorozy/Turkish_sentiment")
ds = ds['train'].train_test_split(test_size=0.2, seed=42)
ds_valid_test = ds['test'].train_test_split(test_size=0.5, seed=42)

final_ds = DatasetDict({
    'train': ds['train'],
    'validation': ds_valid_test['train'],
    'test': ds_valid_test['test']
})
# X, y değerlerini çıkar
X_train = final_ds['train']['text']
y_train = final_ds['train']['label']
X_val = final_ds['validation']['text']
y_val = final_ds['validation']['label']

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_val)

print(classification_report(y_val, y_pred))