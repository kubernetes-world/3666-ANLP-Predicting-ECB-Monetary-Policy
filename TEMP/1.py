import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.utils import to_categorical

# Example DataFrame
df = pd.DataFrame({
    "extracted_text": [
        "The quick brown fox jumps over the lazy dog.",
        "This is an example sentence for testing.",
        "Keyphrases and tokenization for machine learning.",
        "Natural Language Processing is exciting!",
        "Text classification with deep learning models is fascinating."
    ],
    "direction": ["A", "B", "A", "C", "B"]
})

# Custom Transformer: KeyphraseExtractor
class KeyphraseExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=10000, max_length=100):
        self.max_features = max_features
        self.max_length = max_length
        self.tokenizer = Tokenizer(num_words=self.max_features)

    def fit(self, X, y=None):
        """Fit the tokenizer on the text data."""
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X):
        """Transform text data into padded sequences."""
        sequences = self.tokenizer.texts_to_sequences(X)
        return pad_sequences(sequences, maxlen=self.max_length)

# LSTM Model Builder for KerasClassifier
def create_model(vocab_size=10000, input_length=100, num_classes=3):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Encode labels
labels = df["direction"].values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert to one-hot encoding
num_classes = len(label_encoder.classes_)
y = to_categorical(encoded_labels, num_classes=num_classes)

# Define the Pipeline
pipeline = Pipeline([
    ("keyphrase_extractor", KeyphraseExtractor(max_features=10000, max_length=100)),
    ("lstm_classifier", KerasClassifier(build_fn=create_model, vocab_size=10000, input_length=100, num_classes=num_classes, epochs=10, batch_size=32, verbose=0))
])

# Cross-validation setup
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Convert one-hot labels back to single class indices for StratifiedKFold
y_indices = np.argmax(y, axis=1)

# Perform cross-validation
scores = cross_val_score(pipeline, df["extracted_text"], y_indices, cv=skf, scoring="accuracy")

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {np.mean(scores):.2f}")
