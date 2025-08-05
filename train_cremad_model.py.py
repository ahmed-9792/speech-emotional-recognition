import joblib
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Step 1: Enhanced Feature Extraction (MFCC, Deltas, Delta-Deltas, Aggregated Statistics)
def extract_features(file_path, sr=16000, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_features = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1), np.min(mfcc, axis=1), np.max(mfcc, axis=1)])
    delta_features = np.concatenate([np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1), np.min(mfcc_delta, axis=1), np.max(mfcc_delta, axis=1)])
    delta2_features = np.concatenate([np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1), np.min(mfcc_delta2, axis=1), np.max(mfcc_delta2, axis=1)])

    features = np.concatenate([mfcc_features, delta_features, delta2_features])
    return features

# Step 2: Prepare the dataset
data = []
labels = []

emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5
}

# Path to the folder containing .wav files
directory = "C:/Users/Test1/Desktop/SER/AudioWAV"  # Update to your directory

# Iterate over each .wav file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        file_path = os.path.join(directory, filename)

        print(f"Processing file: {filename}")

        feature = extract_features(file_path)
        if feature is None:
            continue

        try:
            emotion_code = filename.split('_')[2].lower()
            print(f"Extracted emotion code: {emotion_code}")

            if emotion_code == 'ang':
                emotion = "angry"
            elif emotion_code == 'dis':
                emotion = "disgust"
            elif emotion_code == 'fea':
                emotion = "fear"
            elif emotion_code == 'hap':
                emotion = "happy"
            elif emotion_code == 'neu':
                emotion = "neutral"
            elif emotion_code == 'sad':
                emotion = "sad"
            else:
                print(f"Unknown emotion code: {emotion_code}")
                continue

            if emotion in emotion_map:
                data.append(feature)
                labels.append(emotion_map[emotion])
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

print(f"Total audio files found: {len(os.listdir(directory))}")
print(f"Total features extracted: {len(data)}")

if len(data) > 0:
    X = np.array(data)
    y = np.array(labels)

    # Step 3: Preprocessing: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify to maintain class proportions

    # Step 4: Feature Scaling (Normalization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for 1D CNN (samples, time_steps, features) - here time_steps = 1 as we aggregated
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)

    # Step 5: Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Step 7: Build the Model (1D CNN)
    model = Sequential([
        Conv1D(128, kernel_size=5, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(256, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(emotion_map), activation='softmax')
    ])

    # Step 8: Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Step 9: Train the model
    epochs = 50
    batch_size = 32
    history = model.fit(X_train_cnn, y_train_encoded, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test_cnn, y_test_encoded))

    # Step 10: Evaluate the model
    score = model.evaluate(X_test_cnn, y_test_encoded, verbose=0)
    print(f"\nTest Loss: {score[0]:.4f}")
    print(f"Test Accuracy: {score[1]:.4f}")

    # Predict on test data
    y_pred_probs = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Generate classification report
    print("\nClassification Report:")
    report = classification_report(y_test_encoded, y_pred, target_names=list(emotion_map.keys()))
    print(report)

    # Save the classification report to a text file
    with open("classification_report.txt", "w") as f:
        f.write(report)

    # Save the classification report to a JSON file
    report_dict = classification_report(y_test_encoded, y_pred, target_names=list(emotion_map.keys()), output_dict=True)
    with open("classification_report.json", "w") as json_file:
        json.dump(report_dict, json_file, indent=4)

    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(emotion_map.keys()),
                yticklabels=list(emotion_map.keys()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

    # Step 11: Save the model
    model.save("speech_emotion_recognition_cnn_model.h5")
    joblib.dump(scaler, 'feature_scaler.pkl')
    print("Trained CNN model and feature scaler saved.")

else:
    print("No valid .wav files found or no valid emotion labels.")
