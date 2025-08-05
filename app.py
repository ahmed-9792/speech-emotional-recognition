import os
from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
from tensorflow.keras.models import load_model # Used for Keras models (.h5)
import joblib # Used for scikit-learn models and vectorizers (.pkl)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import re
import string
from collections import Counter

# Initialize Flask app
app = Flask(__name__)

# Emotion labels (common for both audio and text models)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# --- Load Audio Emotion Model ---
try:
    model = load_model('speech_emotion_recognition_cnn_model.h5')
    print("Audio emotion model loaded successfully!")
except Exception as e:
    print(f"Error loading the audio emotion model: {e}")
    exit() # Critical for audio functionality

# Load the feature scaler for audio
try:
    scaler: StandardScaler = joblib.load('feature_scaler.pkl')
    print("Audio feature scaler loaded successfully!")
except Exception as e:
    print(f"Error loading the audio feature scaler: {e}")
    exit() # Critical for audio functionality

# --- Load Text Emotion Model and Vectorizer ---
text_model = None
text_vectorizer = None

try:
    text_model = joblib.load('text_emotion_model.pkl')
    print("Text emotion model loaded successfully!")
except Exception as e:
    print(f"Error loading text emotion model: {e}")

try:
    text_vectorizer = joblib.load('text_vectorizer.pkl')
    print("Text vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading text vectorizer: {e}")


# Ensure the 'uploads' directory exists inside the 'static' folder
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to extract features for audio prediction
def extract_features_for_prediction(file_path, sr=16000, n_mfcc=13):
    """
    Extracts MFCC features from an audio file for emotion prediction.
    Returns:
        np.array: Concatenated MFCC features, or None if an error occurs.
    """
    y = None # Initialize y to None
    current_sr = sr # Use the default or specified sampling rate

    print(f"DEBUG: Attempting to load audio file: {file_path}")
    try:
        y, current_sr = librosa.load(file_path, sr=sr)
        print(f"DEBUG: librosa.load successful. Audio shape: {y.shape}, Sample rate: {current_sr}")
        if y.size == 0:
            print("DEBUG: Loaded audio has zero length. Cannot extract features.")
            return None
    except Exception as e:
        print(f"DEBUG: Error during librosa.load for {file_path}: {e}")
        return None # Return None on error during loading

    print("DEBUG: Attempting to compute features...")
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=current_sr, n_mfcc=n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_features = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.min(mfcc, axis=1), np.max(mfcc, axis=1)])
        delta_features = np.concatenate([np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1), np.min(mfcc_delta, axis=1), np.max(mfcc_delta, axis=1)])
        delta2_features = np.concatenate([np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1), np.min(mfcc_delta2, axis=1), np.max(mfcc_delta2, axis=1)])

        features = np.concatenate([mfcc_features, delta_features, delta2_features])
        print(f"DEBUG: Feature computation successful. Features shape: {features.shape}")
        return features
    except Exception as e:
        print(f"DEBUG: Error during feature computation for {file_path}: {e}")
        return None


# Function to predict the emotion of a single audio file
def predict_emotion(audio_file_path, scaler: StandardScaler):
    """
    Predicts emotion from an audio file using the loaded audio model.
    Returns:
        tuple: Predicted emotion (str) and probability array (np.array),
               or an error message (str) and None.
    """
    error_message_prefix = "Error: Audio analysis failed."

    # Check if file exists before attempting to load
    if not os.path.exists(audio_file_path):
        return f"{error_message_prefix} Audio file not found at {audio_file_path}.", None

    features = extract_features_for_prediction(audio_file_path)

    # DIAGNOSTIC PRINT: Check the type and value of features
    print(f"DEBUG: Type of 'features' after extraction in predict_emotion: {type(features)}")
    # print(f"DEBUG: Value of 'features' after extraction in predict_emotion: {features}") # Commented out to avoid large output

    if features is None: # This check MUST catch None if feature extraction failed
        return f"{error_message_prefix} Could not extract valid features from audio. Please ensure the audio is not corrupted or too short.", None

    # Ensure features is a numpy array before reshaping
    if not isinstance(features, np.ndarray):
        return f"{error_message_prefix} Unexpected feature format. Expected numpy array, got {type(features)}.", None

    # Reshape for scaling: scaler expects 2D array (n_samples, n_features)
    # This is the line that caused the original AttributeError
    features = features.reshape(1, -1)

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(features)

    # Reshape for the CNN model: model expects 3D input (batch, features, 1)
    scaled_features_cnn = np.expand_dims(scaled_features, axis=-1)

    try:
        # Make prediction
        predictions = model.predict(scaled_features_cnn)

        # Get the index of the emotion with the highest probability
        predicted_class = np.argmax(predictions)

        # Map the index to the emotion label
        predicted_emotion = emotion_labels[predicted_class]

        return predicted_emotion, predictions[0] # Return the predicted label and the probabilities

    except Exception as e:
        return f"{error_message_prefix} During prediction: {e}", None

# Text emotion analysis function using a pre-trained ML model
def analyze_text_emotion(text, text_vectorizer, text_model):
    """
    Analyzes the emotion of a given text using a pre-trained machine learning model.
    Returns:
        tuple: A tuple containing the predicted emotion (str) and a numpy array
               of probabilities for each emotion, or an error message and None.
    """
    if text_vectorizer is None or text_model is None:
        return "Error: Text emotion model or vectorizer not loaded on the server. Please ensure the model files are present.", None

    # Text preprocessing (must match preprocessing used during model training)
    processed_text = str(text).lower()
    processed_text = re.sub(f"[{re.escape(string.punctuation)}]", "", processed_text)
    processed_text = re.sub(r"\s+", " ", processed_text).strip()

    # Transform text using the loaded vectorizer
    try:
        text_features = text_vectorizer.transform([processed_text])
    except Exception as e:
        print(f"Error transforming text features: {e}")
        return "Error: Text feature transformation failed. Check if vectorizer is compatible.", None

    try:
        full_probabilities = np.zeros(len(emotion_labels))

        if hasattr(text_model, 'predict_proba'):
            model_probabilities = text_model.predict_proba(text_features)[0]
            model_classes = text_model.classes_

            for i, class_label in enumerate(model_classes):
                try:
                    global_label_index = emotion_labels.index(class_label)
                    full_probabilities[global_label_index] = model_probabilities[i]
                except ValueError:
                    print(f"Warning: Model predicted class '{class_label}' not in global emotion_labels. Skipping.")
            
            predicted_class_index = np.argmax(full_probabilities)
            predicted_emotion = emotion_labels[predicted_class_index]
            probabilities = full_probabilities

        else:
            predicted_raw_label = text_model.predict(text_features)[0]
            
            try:
                predicted_class_index = emotion_labels.index(predicted_raw_label)
            except ValueError:
                print(f"Warning: Predicted raw label '{predicted_raw_label}' not found in emotion_labels. Defaulting to neutral.")
                predicted_class_index = emotion_labels.index('neutral')

            predicted_emotion = emotion_labels[predicted_class_index]

            print("Warning: Text model does not have 'predict_proba'. Estimating probabilities for chart.")
            probabilities = np.array([1.0 / len(emotion_labels)] * len(emotion_labels))
            probabilities[predicted_class_index] = 0.5 + (1.0 / len(emotion_labels))
            probabilities = probabilities / np.sum(probabilities)

        return predicted_emotion, probabilities
    except Exception as e:
        print(f"Error during text emotion prediction: {e}")
        return f"Error during text emotion prediction: {e}", None

# Route for handling the index page and file upload
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'text_input' in request.form:
            text_input = request.form['text_input'].strip()
            if text_input:
                prediction, probabilities = analyze_text_emotion(text_input, text_vectorizer, text_model)
                
                if isinstance(prediction, str) and prediction.startswith("Error"):
                    return render_template("index.html", error=prediction)
                else:
                    plt.figure(figsize=(6, 4))
                    plt.bar(emotion_labels, probabilities, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3'])
                    plt.xlabel('Emotion')
                    plt.ylabel('Probability')
                    plt.title('Text Emotion Analysis Results')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    img_io = io.BytesIO()
                    plt.savefig(img_io, format='png')
                    img_io.seek(0)
                    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
                    plt.close()

                    return render_template("index.html", 
                                           text_prediction=prediction, 
                                           text_probabilities=probabilities, 
                                           emotion_labels=emotion_labels, 
                                           text_img_base64=img_base64,
                                           analyzed_text=text_input)
            else:
                return render_template("index.html", error="Please enter some text to analyze.")
        
        elif 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            if audio_file and audio_file.filename != '':
                audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
                audio_file.save(audio_path)

                prediction, probabilities = predict_emotion(audio_path, scaler)

                if isinstance(prediction, str) and prediction.startswith("Error"):
                    return render_template("index.html", error=prediction)
                else:
                    plt.figure(figsize=(6, 4))
                    plt.bar(emotion_labels, probabilities, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3'])
                    plt.xlabel('Emotion')
                    plt.ylabel('Probability')
                    plt.title('Audio Emotion Analysis Results')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    img_io = io.BytesIO()
                    plt.savefig(img_io, format='png')
                    img_io.seek(0)
                    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
                    plt.close()

                    return render_template("index.html", prediction=prediction, probabilities=probabilities, 
                                           emotion_labels=emotion_labels, img_base64=img_base64, 
                                           audio_file_path=os.path.relpath(audio_path, UPLOAD_FOLDER))
            else:
                return render_template("index.html", error="No audio file selected for upload.")

        elif 'recorded_audio' in request.files:
            audio_file = request.files['recorded_audio']
            if audio_file and audio_file.filename != '':
                audio_path = os.path.join(UPLOAD_FOLDER, 'recorded_audio.wav')
                audio_file.save(audio_path)

                prediction, probabilities = predict_emotion(audio_path, scaler)

                if isinstance(prediction, str) and prediction.startswith("Error"):
                    return render_template("index.html", error=prediction)
                else:
                    plt.figure(figsize=(6, 4))
                    plt.bar(emotion_labels, probabilities, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3'])
                    plt.xlabel('Emotion')
                    plt.ylabel('Probability')
                    plt.title('Audio Emotion Analysis Results')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    img_io = io.BytesIO()
                    plt.savefig(img_io, format='png')
                    img_io.seek(0)
                    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
                    plt.close()

                    return render_template("index.html", prediction=prediction, probabilities=probabilities, 
                                           emotion_labels=emotion_labels, img_base64=img_base64, 
                                           audio_file_path=os.path.relpath(audio_path, UPLOAD_FOLDER))
            else:
                return render_template("index.html", error="No recorded audio received.")

    return render_template("index.html", error=None)

if __name__ == "__main__":
    app.run(debug=True)
