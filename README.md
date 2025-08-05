Speech Emotion Recognition System
This project is a Flask-based web application that detects human emotions using either audio (speech) or textual input. It integrates two machine learning models:

A 1D CNN model for Speech Emotion Recognition (SER)

A Logistic Regression model for Text Emotion Classification

Both models are trained and deployed to classify emotions into the following categories:

angry, disgust, fear, happy, neutral, sad

📁 Project Structure:

├── app.py                          # Flask web application
├── text_emotion_model_training.py # Script to train text emotion classifier
├── train_cremad_model.py.py       # Script to train audio CNN model (CREMA-D)
├── tweet_emotions.csv             # Dataset for training text emotion model
├── speech_emotion_recognition_cnn_model.h5  # Trained audio model (saved after training)
├── text_emotion_model.pkl         # Trained text model
├── text_vectorizer.pkl            # TF-IDF vectorizer for text preprocessing
├── feature_scaler.pkl             # Scaler for audio features
├── static/
│   └── uploads/                   # Uploaded or recorded audio files
└── templates/
    └── index.html                 # Frontend UI for emotion prediction
    
🚀 Getting Started:

1. Clone the Repository

git clone https://github.com/yourusername/emotion-recognition-app.git
cd emotion-recognition-app

2. Install Dependencies
Create a virtual environment (optional but recommended), then:

pip install -r requirements.txt
Main dependencies:

Flask

TensorFlow

scikit-learn

librosa

matplotlib

seaborn

pandas

🧠 Model Training:

🔹 Text Emotion Model
Uses a cleaned version of tweet_emotions.csv. Emotions are mapped and reduced to 6 core categories.

python text_emotion_model_training.py
This saves:

text_emotion_model.pkl

text_vectorizer.pkl

🔹 Audio Emotion Model
Uses the CREMA-D dataset (update path in train_cremad_model.py.py).

python train_cremad_model.py.py
This saves:

speech_emotion_recognition_cnn_model.h5

feature_scaler.pkl

🌐 Run the Web App:

python app.py
Visit http://127.0.0.1:5000 in your browser.

Features:
Enter a text sentence to classify its emotion

Upload or record audio (.wav) to detect spoken emotion

Visualization of predicted emotion probabilities as bar charts

🎓 Datasets Used:

Text: Custom CSV file tweet_emotions.csv (from Emotion Detection Dataset)

Audio: CREMA-D Dataset

Note: Audio training script assumes filenames contain emotion codes like _ANG, _HAP, etc.

📊 Emotions Supported:

Label	Description
angry	Expressing anger or frustration
disgust	Expressing aversion or disapproval
fear	Expressing worry, fear, or anxiety
happy	Expressing joy, enthusiasm, or love
neutral	Neutral or emotionally flat speech
sad	Expressing sadness or disappointment

📌 Notes:

Make sure to update directory path in train_cremad_model.py.py to your local CREMA-D .wav folder.

Ensure all model files (.h5, .pkl) are in the same directory as app.py or adjust paths accordingly.

Only .wav files are supported for audio input.

📷 Screenshots (Optional):

Include a few screenshots of the UI after predictions (text and audio).

🛠 Future Improvements:

Add support for real-time audio recording in the browser.

Integrate attention-based models for text classification.

Use transfer learning (e.g., Wav2Vec or BERT).
