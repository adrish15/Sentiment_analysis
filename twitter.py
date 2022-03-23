from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
# Load model
model = load_model('adrish_model.h5')
#loading tokenizer
with open('tokenizer.pickle', 'rb') as tokenizer_sav:
    tokenizer = pickle.load(tokenizer_sav)
    
def predict_class(text):
    
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len=600
    
   
    xt = tokenizer.texts_to_sequences(text)

    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)

    print('The predicted sentiment is', sentiment_classes[yt[0]])
    
predict_class(["My experience so far has been fantastic"])

predict_class(["the movie was bad"])

predict_class(["take an umbrella"])
