# Speech-Emotioin-recognition-analysis
 The project aims to develop a machine learning model to identify emotions expressed in speech audio. By leveraging Long Short-Term Memory (LSTM) networks, the model classifies audio recordings into emotional categories such as Positive, Negative, and Neutral.
Problem Statement
Title: Speech Emotion Recognition Using LSTM

Overview: In various applications such as customer service, mental health assessment, and interactive systems, understanding human emotions expressed through speech is crucial. Traditional methods of sentiment analysis focus primarily on text data, neglecting the rich emotional context that voice inflections and tones provide. This project aims to develop a model that accurately classifies emotions in speech audio recordings.

Objective: The primary objective of this project is to design and implement a Long Short-Term Memory (LSTM) neural network model to recognize and classify emotions in spoken language. The model will categorize emotions into three classes: Positive, Negative, and Neutral, based on audio features extracted from the recordings.
CODE EXPLANATION:
!pip install kaggle
Purpose: Installs the Kaggle API, which allows you to interact with Kaggle datasets and competitions directly from your notebook.
Setup Kaggle API Credentials

!mkdir -p ~/.kaggle
Purpose: Creates a directory called .kaggle in your home directory if it doesn't already exist. This is where you will store your Kaggle API credentials.

!cp kaggle.json ~/.kaggle/
Purpose: Copies your kaggle.json file (which contains your API credentials) to the .kaggle directory.

!chmod 600 ~/.kaggle/kaggle.json
Purpose: Changes the permissions of the kaggle.json file to make it readable and writable only by the user. This is a security measure to protect your credentials.

Download Dataset from Kaggle
!kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
Purpose: Downloads the specified dataset (Toronto Emotional Speech Set) from Kaggle.

Unzip the Dataset
!unzip toronto-emotional-speech-set-tess.zip 
Purpose: Unzips the downloaded dataset into the current working directory, making the files accessible for further processing.

Import Libraries
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')
Purpose: Imports necessary libraries for data manipulation (Pandas, NumPy), visualization (Seaborn, Matplotlib), audio processing (Librosa), and to suppress warnings.

Initialize Lists for Paths and Labels
paths = []
labels = []
Purpose: Initializes empty lists to store the paths of audio files and their corresponding labels (emotions).
Set Dataset Directory

dataset_directory = '/content/tess toronto emotional speech set data/TESS Toronto emotional speech set data'
Purpose: Defines the path to the dataset directory, which will be used to locate audio files.
Traverse Dataset Directory

for dirname, _, filenames in os.walk(dataset_directory):
Purpose: Walks through the directory structure, allowing you to access all files within the specified dataset directory.

for filename in filenames:
    paths.append(os.path.join(dirname, filename))
    label = filename.split('_')[-1]
    label = label.split('.')[0]
    labels.append(label.lower())
    if len(paths) == 2800:  # Limit to 2800 samples, if needed
        break
Purpose: For each file in the dataset:
Adds the full path to the paths list.
Extracts the label (emotion) from the filename, assuming the format includes the emotion as part of the name.
Converts the label to lowercase and appends it to the labels list.
Stops collecting paths after reaching 2800 samples, if specified.

print('Dataset is Loaded')
Purpose: Prints a message indicating that the dataset has been successfully loaded.

len(paths)
Purpose: Returns the total number of audio files loaded into the paths list.

labels[:5]
Purpose: Displays the first five labels to verify the loading process.

Create a DataFrame
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()
Purpose: Creates a Pandas DataFrame containing the paths and labels. The head() method displays the first few rows of the DataFrame.
Visualize Label Distribution

df['label'].value_counts()
Purpose: Counts the occurrences of each emotion label in the DataFrame, allowing you to understand the distribution of the dataset.

sns.countplot(data=df, x='label')
Purpose: Uses Seaborn to create a count plot that visually represents the distribution of different emotion labels in the dataset.
Define Functions for Waveform and Spectrogram Visualization

def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()
Purpose: Defines a function to plot the waveform of an audio signal. Takes the audio data, sampling rate, and emotion label as inputs.

def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
Purpose: Defines a function to plot the spectrogram of an audio signal. Uses Short-Time Fourier Transform (STFT) to convert the audio signal to the frequency domain.
Visualize Different Emotions
For each emotion, the following block of code loads the corresponding audio file, visualizes its waveform and spectrogram, and plays the audio:


emotion = 'fear'
path = np.array(df['speech'][df['label']==emotion])[0]
data, sampling_rate = librosa.load(path)
waveplot(data, sampling_rate, emotion)
spectogram(data, sampling_rate, emotion)
Audio(path)
Purpose: Repeats for different emotions like 'angry', 'disgust', 'neutral', etc., allowing for visual and auditory exploration of the dataset.
Feature Extraction

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc
Purpose: Defines a function to extract MFCC features from audio files. It loads the audio file, computes MFCCs, and returns the mean of these features.

X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
X = [x for x in X_mfcc]
X = np.array(X)
Purpose: Applies the extract_mfcc function to each audio file in the DataFrame to create an array of MFCC features.

X.shape
Purpose: Displays the shape of the extracted feature array, confirming the number of samples and feature dimensions.
Reshape Input Data

X = np.expand_dims(X, -1)
Purpose: Expands the dimensions of X to make it suitable for input into the LSTM model.
One-Hot Encoding of Labels

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])
y = y.toarray()
Purpose: Initializes a OneHotEncoder, fits it to the labels, and transforms the labels into a one-hot encoded format.
Build the LSTM Model

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])
Purpose: Defines a sequential LSTM model architecture:
LSTM Layer: First layer with 256 units.
Dropout Layer: Reduces overfitting by randomly setting 20% of the input units to 0.
Dense Layers: Fully connected layers with ReLU activations to learn complex representations.
Output Layer: Softmax activation to produce probabilities for 7 emotion classes.
Compile the Model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Purpose: Compiles the model with the specified loss function (categorical cross-entropy), optimizer (Adam), and evaluation metric (accuracy).
Model Summary

model.summary()
Purpose: Displays the architecture of the model, including the number of parameters in each layer.
Train the Model

history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)
Purpose: Trains the model on the input features (X) and labels (y) for 50 epochs, using 20% of the data for validation and a batch size of 64.
Plot Training Accuracy and Validation Accuracy

epochs = list(range(50))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(







