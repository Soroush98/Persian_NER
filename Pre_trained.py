import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
def create_model():
    emb_word = Embedding(input_dim=n_words + 2, output_dim=20, input_length=max_len, mask_zero=True)(word_in)
    char_in = Input(shape=(max_len, max_len_char,))
    emb_char = TimeDistributed(
        Embedding(input_dim=n_chars + 2, output_dim=10, input_length=max_len_char, mask_zero=True))(char_in)
    char_enc = TimeDistributed(LSTM(units=20, return_sequences=False, recurrent_dropout=0.5))(emb_char)
    x = concatenate([emb_word, char_enc])
    x = SpatialDropout1D(0.3)(x)
    main_lstm = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.6))(x)
    out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)
    model = Model([word_in, char_in], out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    return model

def load_trained_model(weights_path):
   model = create_model()
   model.load_weights(weights_path)