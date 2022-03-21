import streamlit as st
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

imdb_reviews=pd.read_csv("imdb_reviews.csv")
test_reviews=pd.read_csv("test_reviews.csv")
word_index=pd.read_csv("word_indexes.csv")
word_index=dict(zip(word_index.Words,word_index.Indexes))

word_index["<PAD>"]=0
word_index["<START"]=1
word_index["<UNK>"]=2
word_index["<UNUSED>"]=3

def review_encoder(text):
  arr=[word_index[word] for word in text]
  return arr
train_data,train_labels=imdb_reviews['Reviews'],imdb_reviews['Sentiment']
test_data, test_labels=test_reviews['Reviews'],test_reviews['Sentiment']

train_data=train_data.apply(lambda review:review.split())
test_data=test_data.apply(lambda review:review.split())

train_data=train_data.apply(review_encoder)
test_data=test_data.apply(review_encoder)


def encode_sentiments(x):
  if x=='positive':
    return 1
  else:
    return 0

train_labels=train_labels.apply(encode_sentiments)
test_labels=test_labels.apply(encode_sentiments)


train_data=keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding='post',maxlen=500)
test_data=keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding='post',maxlen=500)

model=keras.Sequential([keras.layers.Embedding(10000,16,input_length=500),
                        keras.layers.GlobalAveragePooling1D(),
                        keras.layers.Dense(16,activation='relu'),
                        keras.layers.Dense(1,activation='sigmoid')])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


history=model.fit(train_data,train_labels,epochs=30,batch_size=512,validation_data=(test_data,test_labels))
index=np.random.randint(1,1000)

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
