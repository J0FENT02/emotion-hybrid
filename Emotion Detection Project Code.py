import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nrclex import NRCLex
from transformers import pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Preprocessing
lemmatizer = WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words('english'))

def preprocess_text(text):
    txt = re.sub(r'http\S+|www\S+|@\w+', '', str(text).lower())
    tokens = word_tokenize(txt)
    return [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stopwords]

# Lexicon features (NRC)
EMO_KEYS = ['joy','sadness','fear','anger','surprise','disgust']
EMO_CATS = ['Happiness','Sadness','Fear','Anger','Surprise','Disgust']

def lexicon_features(text):
    tokens = preprocess_text(text)
    doc = NRCLex(" ".join(tokens))
    raw = doc.raw_emotion_scores  # counts
    counts = [ raw.get(k,0) for k in EMO_KEYS ]
    total = sum(counts) or 1
    return [c/total for c in counts]

# Transformer features
clf_tf = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=-1
)

LABEL_MAP = {
    'joy':'Happiness','sadness':'Sadness','fear':'Fear',
    'anger':'Anger','surprise':'Surprise','disgust':'Disgust'
}

def transformer_features(text):
    out = clf_tf(text)[0]
    d = {item['label'].lower(): item['score'] for item in out}
    return [ d.get(k,0.0) for k in EMO_KEYS ]

# Feature extraction
def extract_features(text):
    return np.array(lexicon_features(text) + transformer_features(text))

# Dataset preparation
if __name__ == "__main__":
    df = pd.read_csv("emotion-dataset.csv")
    # Map ints to emotions
    INT2STR = {0:"Sadness",1:"Happiness",2:"Love",3:"Anger",4:"Fear",5:"Surprise"}
    df['Emotion'] = df['label'].map(INT2STR)
    # Filter out love (no feature for it in NRC)
    df = df[df['Emotion']!='Love'].reset_index(drop=True)
    
    # Feature matrix (x), label vector (y)
    print("Extracting features for", len(df), "samples…")
    X = np.vstack(df['text'].apply(extract_features).values)
    y = df['Emotion'].values

    # 5‑fold cross‑validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000,multi_class="multinomial"))
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"5‑fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Held‑out evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("\n=== Held‑out Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=sorted(set(y))))
    print("Held‑out accuracy:", accuracy_score(y_test, y_pred))
