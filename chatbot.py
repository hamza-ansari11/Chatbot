import os
import nltk 
nltk.download('punkt')
nltk.download('stopwords')

import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context=ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

import json
with open("chatbot_intents.json","r") as file:
    intents=json.load(file)


vectorizer=TfidfVectorizer()
clf=LogisticRegression(random_state=0,max_iter=10000)

tags=[]
patterns=[]
for intent in intents["intents"]:  # Iterate through the list of intents
    for pattern in intent["patterns"]:  # Iterate through patterns within each intent
        tags.append(intent["tag"])  # Append the tag of the current intent
        patterns.append(pattern)  # Append the current pattern

x=vectorizer.fit_transform(patterns)
y=tags
clf.fit(x,y)

def chatbot(input_text):
    input_text=vectorizer.transform([input_text])
    tag=clf.predict(input_text)[0]
    for intent in intents["intents"]:
        if intent['tag']==tag:
            response=random.choice(intent['responses'])
            return response

counter=0
def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome to Chatbot")

    counter+=1
    user_input=st.text_input("You: ",key=f"user_input_{counter}")

    if user_input:
        response=chatbot(user_input)
        st.text_area("chatbot:",value=response,height=100,max_chars=None,key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye','bye']:
            st.write("Thanks")
            st.stop()
if __name__=='__main__':
    main()