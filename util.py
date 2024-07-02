import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

CapGenerator = tf.keras.models.load_model('models/CapGen.h5')
VGGMod = tf.keras.models.load_model('models/VGGModel.h5')
max_length = 35

with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

vocab_size = len(tokenizer.word_index) + 1

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length=max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text
  
def feature_extractor(image):

    # Img to np array
    image = img_to_array(image)

    # Reshaping
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Preprocessing for passing through VGG16
    image = preprocess_input(image)

    feature = VGGMod.predict(image, verbose=0)
    
    return feature

def generate_caption(image_name):
    
    y_pred = predict_caption(CapGenerator, feature_extractor(image_name), tokenizer, max_length)
    y_pred = y_pred[8:-7].upper()
    
    return y_pred
    