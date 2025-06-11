import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Download NLTK stopwords
nltk.download('stopwords')

# Sample training data (text, category)
training_data = [
    ("I can't connect to the internet", "technical support"),
    ("My printer isn't working", "technical support"),
    ("I need help with my account settings", "technical support"),
    ("Where is my order?", "shipping"),
    ("The package hasn’t arrived yet", "shipping"),
    ("I want to track my shipment", "shipping"),
    ("I was charged twice", "billing"),
    ("There is an error in my invoice", "billing"),
    ("Can I change my payment method?", "billing"),
]

# Text preprocessing setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and t not in string.punctuation]
    return ' '.join(tokens)

# Preprocess the training data
texts = [preprocess(text) for text, label in training_data]
labels = [label for text, label in training_data]

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Optional evaluation on test set
print("=== Naive Bayes Evaluation ===")
print(classification_report(y_test, nb_model.predict(X_test)))
print("=== Decision Tree Evaluation ===")
print(classification_report(y_test, dt_model.predict(X_test)))

# Classification function for both models
def classify_with_both_models(user_input):
    cleaned = preprocess(user_input)
    vector = vectorizer.transform([cleaned])
    nb_prediction = nb_model.predict(vector)[0]
    dt_prediction = dt_model.predict(vector)[0]
    return nb_prediction, dt_prediction

# Chatbot function
def chatbot():
    print("🤖 Hello! I can help you with technical support, billing, or shipping issues.")
    print("Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Goodbye! 👋")
            break
        nb_result, dt_result = classify_with_both_models(user_input)
        print(f"🔍 Naive Bayes says: '{nb_result}'")
        print(f"🌳 Decision Tree says: '{dt_result}'")

# Run chatbot
if __name__ == "__main__":
    chatbot()
    