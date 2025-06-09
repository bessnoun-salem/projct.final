import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Download stopwords only (no punkt needed)
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

# Preprocessing setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')  # Regex tokenizer to avoid using 'punkt'

# Preprocessing function
def preprocess(text):
    tokens = tokenizer.tokenize(text.lower())  # Tokenize without punkt
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Preprocess the training data
texts = [preprocess(text) for text, label in training_data]
labels = [label for text, label in training_data]

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, labels)

# Classification function
def classify_user_input(user_input):
    cleaned = preprocess(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = classifier.predict(vector)
    return prediction[0]

# Chatbot loop
def chatbot():
    print("🤖 Hello! I can help you with technical support, billing, or shipping issues.")
    print("Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Goodbye! 👋")
            break
        category = classify_user_input(user_input)
        print(f"Bot: This seems like a '{category}' issue.")

# Run the chatbot
if __name__ == "__main__":
    chatbot()
