import random
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

# Load necessary files
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Clean up user input
def clean_up_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence to bag of words
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

# Predict the class
def predict_class(sentence):
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    results = [(i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

# Get response based on prediction
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I didn't understand that. Can you try again?"
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Chat with bot
def chatbot_response(msg):
    intents_list = predict_class(msg)
    return get_response(intents_list, intents)

# Loop for chatting
print("Chatbot is ready! Type 'quit' to exit.")
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break
    response = chatbot_response(message)
    print("Bot:", response)
