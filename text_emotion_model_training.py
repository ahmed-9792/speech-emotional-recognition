import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import string
import numpy as np

def preprocess_text(text):
    """
    Cleans and preprocesses the input text.
    Steps: lowercase, remove punctuation, remove extra spaces.
    """
    # Ensure text is treated as a string, converting if it's a float (NaN values)
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # Remove extra spaces and strip
    return text

def train_text_emotion_model(data_path="tweet_emotions.csv", text_column="content", label_column="sentiment"):
    """
    Trains a text emotion recognition model and saves it along with the vectorizer.

    Args:
        data_path (str): Path to the CSV dataset file.
        text_column (str): Name of the column containing the text data.
        label_column (str): Name of the column containing the emotion labels.
    """
    print(f"Loading dataset from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Dataset '{data_path}' not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- DEBUGGING AID: Print all column names after loading ---
    print(f"\nColumns in the loaded dataset: {df.columns.tolist()}")
    # --- END DEBUGGING AID ---

    # Display basic info about the dataset
    print("\nDataset Info:")
    df.info()
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print(f"\nOriginal unique emotion labels from '{label_column}' column: {df[label_column].unique()}")
    print(f"Original emotion distribution from '{label_column}' column:\n{df[label_column].value_counts()}")

    # Define the mapping from dataset sentiments to target emotions
    sentiment_mapping = {
        'sadness': 'sad',
        'enthusiasm': 'happy',
        'worry': 'fear', # Mapping worry to fear as it implies apprehension
        'love': 'happy',
        'surprise': 'neutral', # Surprise can be positive or negative, mapping to neutral for simplicity
        'fun': 'happy',
        'hate': 'angry',
        'happiness': 'happy',
        'boredom': 'sad',
        'relief': 'happy',
        'anger': 'angry',
        'empty': 'neutral' # Tweets marked 'empty' often lack strong sentiment, hence neutral
    }
    
    # Apply mapping and filter out unmapped sentiments
    df['mapped_sentiment'] = df[label_column].map(sentiment_mapping)
    df.dropna(subset=['mapped_sentiment'], inplace=True) # Drop rows where mapping resulted in NaN
    df.rename(columns={'mapped_sentiment': 'emotion'}, inplace=True) # Rename the new column to 'emotion'

    # Filter for the target emotions (optional, if mapping covers all)
    # Ensure the final labels match app.py's emotion_labels
    target_emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    df = df[df['emotion'].isin(target_emotion_labels)]

    print(f"\nMapped unique emotion labels: {df['emotion'].unique()}")
    print(f"Mapped emotion distribution:\n{df['emotion'].value_counts()}")

    # Preprocess the text data
    print("\nPreprocessing text data...")
    df['cleaned_text'] = df[text_column].apply(preprocess_text)
    # Drop rows where cleaned_text might become empty after preprocessing
    df.dropna(subset=['cleaned_text'], inplace=True)
    df = df[df['cleaned_text'].str.strip() != ''] # Remove rows with empty strings after cleaning
    print("Text preprocessing complete.")
    print(f"Dataset size after preprocessing and filtering: {len(df)} samples.")

    if len(df) == 0:
        print("Error: No valid data left after preprocessing and mapping. Cannot train model.")
        return

    # Split data into training and testing sets
    X = df['cleaned_text']
    y = df['emotion']

    # Ensure all target emotions are present in both train and test sets if stratify is used
    class_counts = y.value_counts()
    # Check if any class has less than 2 samples (required for stratified split)
    if any(count < 2 for count in class_counts):
        print("\nWarning: Some emotion classes have very few samples (<2). Stratified split might fail or result in empty classes.")
        print("Falling back to non-stratified split for this run.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nData split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

    # Initialize and train TF-IDF Vectorizer
    print("\nTraining TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.8)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    print(f"TF-IDF Vectorizer trained. Number of features: {X_train_vectorized.shape[1]}")

    # Initialize and train a Logistic Regression Classifier
    print("\nTraining Logistic Regression Model...")
    model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
    model.fit(X_train_vectorized, y_train)
    print("Model training complete.")

    # Evaluate the model
    print("\nEvaluating the model...")
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    # Get unique labels from the test set for the classification report
    unique_labels_test = np.unique(y_test).tolist()
    print(classification_report(y_test, y_pred, target_names=unique_labels_test, zero_division='warn'))

    # Save the trained model and vectorizer
    model_filename = 'text_emotion_model.pkl'
    vectorizer_filename = 'text_vectorizer.pkl'

    try:
        joblib.dump(model, model_filename)
        print(f"\nModel saved successfully as '{model_filename}'")
        joblib.dump(vectorizer, vectorizer_filename)
        print(f"Vectorizer saved successfully as '{vectorizer_filename}'")
        print("\nYou can now use these files in your Flask app for text emotion analysis.")
    except Exception as e:
        print(f"Error saving model or vectorizer: {e}")

if __name__ == "__main__":
    # Call the training function with your dataset details
    # Correctly specifying text_column and label_column for tweet_emotions.csv
    train_text_emotion_model(data_path="tweet_emotions.csv", text_column="content", label_column="sentiment")

