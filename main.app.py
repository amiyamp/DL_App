import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import numpy as np
from keras.preprocessing import image as keras_image
from PIL import Image
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence



# Load your models
rnn_model = load_model('spam_model.h5')
lstm_model = load_model('LSTM_model.h5')
#cnn_model = tf.keras.models.load_model('tumor_model.h5')
dnn_model = load_model('spam_dnn_model.h5')


with open(r'perceptron_model.pkl', 'rb') as handle:
    perceptron_model = pickle.load(handle)
with open(r'backpropagation_model.pkl', 'rb') as handle:
    backpropagation_model = pickle.load(handle)

with open(r'tokeniser.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Streamlit app
st.title('Sentiment & Image Classification')

# Radio box for choosing text or image classification
task = st.radio('Choose a task:', ('Sentiment Classification', 'Tumor Classification'))

if task == 'Sentiment Classification':
    # Dropdown box for choosing the text classification model
    text_model = st.selectbox('Select Text Classification Model:', ('RNN (SpamSMS)', 'LSTM (IMDb)', 'DNN (SpamSMS)','Perceptron (SpamSMS)','Backpropagation (SpamSMS)'))

    # Sample input for text classification
    text_input = st.text_input('Enter text for classification:', 'Your sample text here.')

    # Button to trigger prediction
    if st.button('Predict'):
        # Perform classification based on the selected model
        if text_model == 'RNN (SpamSMS)':
            text_sequence = tokenizer.texts_to_sequences([text_input])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(text_sequence, maxlen=10, padding="post")
            # Make predictions using rnn_model on padded_sequence
            prediction = rnn_model.predict(padded_sequence)
           
            # Display the result for the RNN model
            result = "Spam" if prediction >0.3 else "Ham"
            
        elif text_model == 'DNN (SpamSMS)':
            # Tokenize and pad the input text
            text_sequence = tokenizer.texts_to_sequences([text_input])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(text_sequence, maxlen=10, padding="post")
            # Make predictions using dnn_model on padded_sequence
            prediction = dnn_model.predict(padded_sequence)
            result = "Spam" if prediction > 0.3 else "Ham"
            
        elif text_model == 'Perceptron (SpamSMS)':
            # Tokenize and pad the input text
            text_sequence = tokenizer.texts_to_sequences([text_input])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(text_sequence, maxlen=10, padding="post")
            # Make predictions using dnn_model on padded_sequence
            prediction = dnn_model.predict(padded_sequence)
            result = "Spam" if prediction > 0.3 else "Ham"
            

        elif text_model == 'Backpropagation (SpamSMS)':
            # Tokenize and pad the input text
            text_sequence = tokenizer.texts_to_sequences([text_input])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(text_sequence, maxlen=10, padding="post")
            # Make predictions using backpropagation_model on padded_sequence
            prediction = dnn_model.predict(padded_sequence)
            print("Raw Predictions:", prediction)
            result = "Spam" if prediction> 0.3 else "Ham"

            


        elif text_model == 'LSTM (IMDb)':
                words = 5000
                max_review_length=500
                word_index = imdb.get_word_index()
                text_input = text_input.lower().split()
                text_input = [word_index[word] if word in word_index and word_index[word] < words else 0 for word in text_input]
                text_input = sequence.pad_sequences([text_input], maxlen=max_review_length)
                prediction = lstm_model.predict(text_input)
                print("Raw Prediction:", prediction)
                result = "Positive" if prediction > 0.5  else "Negative"
                

        st.write('Prediction:', result)

elif task == 'Tumor Classification':
    # File uploader for image classification
    uploaded_file = st.file_uploader('Choose an image for classification:', type=['jpg', 'jpeg', 'png'])

    # Button to trigger prediction
    if st.button('Predict'):
        # Perform classification based on the CNN model
        if uploaded_file is not None:
            # Preprocess the image
            image = Image.open(uploaded_file)
            # Perform any necessary preprocessing for your CNN model
            image = image.resize((128, 128))  # Adjust the size based on your model's requirements
            image_array = keras_image.img_to_array(image)

            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Make predictions using cnn_model on the preprocessed image
            prediction = cnn_model.predict(image_array)

            # Make predictions using cnn_model on the preprocessed image
            result = "Tumor Detected" if prediction > 0.5 else "No Tumor"
            
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write('Prediction:',result)
            

