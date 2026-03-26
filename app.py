import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

download_nltk_data()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'vectorizer.pkl' or 'model.pkl' not found in the current directory.")
st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message below to analyze:", height=150)

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT SPAM (Ham)")