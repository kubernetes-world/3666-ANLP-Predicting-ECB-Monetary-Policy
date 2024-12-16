X_ = pd.Series(np.random.randn(10))  # Simulating a 1D pandas Series
y_ = np.concatenate([np.zeros(7), np.ones(3)])  # Simulating an imbalanced numpy array

display(X_)
display(y_)

print(80*'-')

X_ = X_.values.reshape(-1, 1)
y_ = y_.reshape(-1,)

display(X_)
display(y_)

smote = SMOTE(k_neighbors=2, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_, y_)

# #####


keyphrase_extractor = KeyphraseExtractor(nfeatures=10000, doclen=100)
X_train_transformed = keyphrase_extractor.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
