import streamlit as st
import pandas as pd
import pickle as pkl
import string
import nltk
nltk.download("punkt")
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer


model = pkl.load(open('model.pkl', 'rb'))
tf = pkl.load(open('tf.pkl', 'rb'))


def text_transformer(col):
    stemmer = PorterStemmer()
    return col.str.lower().str.translate(str.maketrans('', '', string.punctuation)).apply(nltk.word_tokenize).apply(lambda x: ' '.join([stemmer.stem(word) for word in x]))


def pred(text):
    text = tf.transform([text]).toarray()
    return 'Ham 😇' if model.predict(text)[0] == 0 else 'Likely Spam ⚠️'


def set_bg_color(color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


st.title('Is your message ham or spam?')
st.write('*Spam detection using machine learning.*')
st.caption("*-by Md Faisal.*")

uploaded_file = st.sidebar.file_uploader('Upload your file', type=['csv'])

for i in range(3):
    st.write("")
msg = st.text_input('Please enter the message.')

if st.button('Enter'):
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)

            text_col = df.select_dtypes(include=[object]).columns[0]
            df[text_col] = text_transformer(df[text_col])
            predictions = [pred(text) for text in df[text_col]]

            df['prediction'] = predictions
            st.write(df[[text_col, 'prediction']])

            if 'Likely Spam ⚠️' in predictions:
                set_bg_color('linear-gradient(to bottom, #ff4e50, #f9d423)')
        else:
            st.error('File format not supported. Please upload a CSV file.')
    elif msg:
        special_users = ['nikhilxhitler', 'kalir', 'cnu', 'arvindgand','irfandon','prasadtopper','sandeeptopper']
        if msg.lower() in special_users:
            st.image('https://i.pinimg.com/564x/4c/6b/19/4c6b19556450b88fe42d89a31dfc033d.jpg')
        else:
            prediction = pred(msg)
            st.write(prediction)
            if 'Likely Spam ⚠️' in prediction:
                set_bg_color('linear-gradient(to bottom, #ff4e50, #f9d423)')
    else:
        st.error('Please upload a file or enter a message.')

for i in range(20):
    st.write("")

st.divider()
st.subheader("Example Messages")
example_messages = ["Congratulations! You've won a $1000 gift card.",
                    "Don't forget our meeting, tomorrow at 10 AM."]
for message in example_messages:
    st.write(f"Message  :  {message}   -   Prediction: {pred(message)}")

st.subheader("Tips for users")
st.write("""
- Please click on > in the top right corner to upload any CSV file if you are using it on your phone.
- Ensure your CSV file has a text column for the messages.
- Enter individual messages for quick predictions.
- Check out my profiles and resume for more information about my work!
""")

# Personal links
for i in range(20):
    st.sidebar.write("")

st.sidebar.divider()
st.sidebar.header("Connect with me")
st.sidebar.markdown("""
<a href="https://www.linkedin.com/in/md-fsl?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="30" height="30">
</a> &nbsp; &nbsp;
<a href="https://github.com/Muhammed-Faisal" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="30" height="30">
</a> &nbsp; &nbsp;
<a href="https://www.kaggle.com/mdfaisal1" target="_blank">
<img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-1024.png" width="30" height="30">
</a>
""", unsafe_allow_html=True)
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.header("Have any suggestions?")
st.sidebar.write("*Do let me know at mdf1234786143@gmail.com*")
