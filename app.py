import streamlit as st
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import numpy as np

from keras.models import load_model
model = load_model('my_model.h5')

def encode_sentiments(x):
  if x=='positive':
    return 1
  else:
    return 0

st.title("Welcome to Sentiment analyzer")
user_review = st.text_input("Your feedback", key="text")
user_review = user_review.split()
user_review = review_encoder(user_review)
user_review=np.array([user_review])
user_review=keras.preprocessing.sequence.pad_sequences(user_review,value=word_index["<PAD>"],padding='post',maxlen=500)

add_selectbox = st.sidebar.selectbox(
    'Feedback category',
    ('Movies', 'Books')
)
#st.text_input("Your feedback", key="text")

left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('sentiment!')

if (model.predict(user_review)>0.5).astype("int32"):
  st.write('positive sentiment')
else:
  st.write("negative sentiment")
