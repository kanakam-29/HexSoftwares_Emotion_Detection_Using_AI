# HexSoftwares_Emotion_Detection_Using_AI
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.metrics import accuracy_score # type: ignore


# Step 1: Create Dataset

data = {
    "text": [
        "I am very happy today",
        "I feel so sad",
        "I am really angry",
        "This is amazing",
        "I am feeling depressed",
        "I love this product",
        "I hate this service",
        "This is wonderful",
        "I feel terrible today",
        "I am excited",
        "I am frustrated",
        "This makes me smile",
        "I feel lonely",
        "This is disappointing",
        "I am very satisfied",
        "I am nervous",
        "I am proud of this",
        "I feel hopeless",
        "I am calm and relaxed",
        "I am very upset"
    ],

    "emotion": [
        "happy",
        "sad",
        "angry",
        "happy",
        "sad",
        "happy",
        "angry",
        "happy",
        "sad",
        "happy",
        "angry",
        "happy",
        "sad",
        "sad",
        "happy",
        "fear",
        "happy",
        "sad",
        "calm",
        "sad"
    ]
}

df = pd.DataFrame(data)


# Step 2: Convert text to numerical data using NLP

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["text"])

y = df["emotion"]


# Step 3: Split dataset into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 4: Train Machine Learning Model

model = MultinomialNB()

model.fit(X_train, y_train)


# Step 5: Test the model

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# Step 6: Take user input

while True:

    user_input = input("\nEnter a sentence (type 'exit' to stop): ")

    if user_input.lower() == "exit":
        print("Program ended.")
        break

    # Convert user input into vector
    user_vector = vectorizer.transform([user_input])

    # Predict emotion
    prediction = model.predict(user_vector)

    print("Predicted Emotion:", prediction[0])
