import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import json
import random
import pickle
from nltk.tokenize import word_tokenize

# Download necessary NLTK packages
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load data from JSON file (make sure you have this file)
with open('intents.json') as file:
    intents = json.load(file)

# Initialize data containers
training_sentences = []
training_labels = []
classes = []
words = []
ignore_words = ['?', '!', '.', ',']

# Loop through each intent
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        # Add the sentence to training set
        training_sentences.append(pattern)
        # Add the associated tag to labels
        training_labels.append(intent['tag'])
    
    # Add the tag to classes if it's not already there
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort the classes
classes = sorted(list(set(classes)))

# Create the training set
training = []
output_empty = [0] * len(classes)

# Create bag of words for each sentence
for i, sentence in enumerate(training_sentences):
    bag = []
    word_list = word_tokenize(sentence)
    word_list = [lemmatizer.lemmatize(w.lower()) for w in word_list]
    
    # Create the bag of words
    for w in words:
        bag.append(1 if w in word_list else 0)
    
    # Output is a 0 for each tag and 1 for the correct tag index
    output_row = list(output_empty)
    output_row[classes.index(training_labels[i])] = 1
    
    # Add the data to the training set
    training.append([bag, output_row])

# Check the structure of the training data
for data in training:
    print(len(data[0]))  # Length of bag of words
    print(len(data[1]))  # Length of output row (one-hot encoding)

# Shuffle the training data
random.shuffle(training)

# Convert to numpy arrays
train_x = np.array([item[0] for item in training])  # Bag of words
train_y = np.array([item[1] for item in training])  # Output row

# Check if lengths match
assert train_x.shape[1] == len(words), "Mismatch in input size"
assert train_y.shape[1] == len(classes), "Mismatch in output size"

print(f"Training data created\n{len(training_sentences)} documents\n{len(classes)} classes\n{len(words)} unique lemmatized words")

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model with updated optimizer
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save("chatbot_model.h5")
print("Model trained and saved!")

# Save the data structures for later use
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
