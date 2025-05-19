# 🩺 Patient Information Chatbot (NLP-Based)

This is a **Natural Language Processing (NLP)** based terminal chatbot that provides **basic support and information** for healthcare-related queries like:
- Adverse Drug Reactions
- Blood Pressure Management
- Hospital and Pharmacy Search

It simulates a hospital assistant helping users find relevant information using a predefined intent-based response system.

---

##  Features

- Responds to greetings, thanks, and farewells
- Helps navigate through:
  - **Adverse Drug Reaction** modules
  - **Blood Pressure tracking and results**
  - **Pharmacy search by name**
  - **Hospital search by name, location, and type**

---

##  How to Run the Project

### 1. Clone the Repository

git clone https://github.com/Anushka0206/chatbot-nlp.git
cd chatbot-nlp
2. Install Required Libraries
bash
Copy
Edit
pip install nltk
- Make sure Python is installed.

3. Run the Chatbot in Terminal
bash
Copy
Edit
python chatbot.py
#### Project Structure
bash
Copy
Edit
chatbot-nlp/
│
├── intents.json         # Contains training data (intents, patterns, responses)
├── chatbot.py           # Main script to run the chatbot
├── words.pkl            # Preprocessed words file
├── classes.pkl          # Preprocessed classes (intents)
├── README.md            # Project documentation
#### Sample Chat
vbnet
Copy
Edit
You: Hello
Bot: Hello, thanks for asking.

You: I want to check my blood pressure history
Bot: Please provide Patient ID

You: Find me a pharmacy
Bot: Please provide pharmacy name
#### Use Cases
College/minor-level AI/NLP project

Healthcare demo for basic hospital chatbot

Foundation for building advanced ML-based virtual assistants

#### Future Improvements
Integrate with actual medical APIs/databases

Add voice input/output

GUI interface using Tkinter or web
