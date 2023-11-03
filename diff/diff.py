import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# define the dataset path and emotion labels
data_path = '\TESS Toronto emotional speech set data'
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# define a function to extract audio features from a single file
def extract_features(file_path):
    # load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # extract the Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)

    # pad the features to a consistent shape
    pad_width = 300 - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mfccs.flatten()

# load the dataset and extract features
X = []
y = []
for label in labels:
    label_path = os.path.join(data_path, label)
    for filename in os.listdir(label_path):
        file_path = os.path.join(label_path, filename)
        features = extract_features(file_path)
        X.append(features)
        y.append(label)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train a support vector machine (SVM) classifier
clf = SVC(kernel='linear', C=1, random_state=42)
clf.fit(X_train, y_train)

# make predictions on a single audio file
audio_path = '/path/to/audio_file'
features = extract_features(audio_path)
emotion = clf.predict([features])[0]
print("Emotion:", emotion)
