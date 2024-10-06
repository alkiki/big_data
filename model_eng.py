import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
# Initialize the tokenizer
tokenizer = Tokenizer()
# Read the file containing the poems
filepath = 'eng_datasets/combined_poems_eng.csv'
data = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
# Split the data into individual lines/poems
corpus = data.lower().split("\n")
# Fit the tokenizer on the corpus to build the word index
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1  # Total unique words + 1 for padding
# Prepare input sequences
input_sequences = []
for line in corpus:
   token_list = tokenizer.texts_to_sequences([line])[0]  # Convert text to sequence of integers
   for i in range(1, len(token_list)):
       n_gram_sequence = token_list[:i+1]  # Create n-gram sequences
       input_sequences.append(n_gram_sequence)
# Pad sequences to ensure they are of the same length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
# Create predictors (xs) and labels (ys)
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)  # One-hot encode the labels
# Build the model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))  # Embedding layer
model.add(Bidirectional(LSTM(150)))  # Bidirectional LSTM layer
model.add(Dense(total_words, activation='softmax'))  # Output layer with softmax activation
adam = Adam(learning_rate=0.01)  # Optimizer
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])  # Compile the model
# Train the model
history = model.fit(xs, ys, epochs=100, verbose=1)
# Seed text for generating new text
seed_text = 'My tired soul, let peace be your reward'
next_words = 100
for _ in range(next_words):
   token_list = tokenizer.texts_to_sequences([seed_text])[0]  # Convert seed text to sequence
   token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')  # Pad the sequence
   predicted = np.argmax(model.predict(token_list), axis=-1)  # Predict the next word
   output_word = ""
   for word, index in tokenizer.word_index.items():  # Find the word corresponding to the predicted index
       if index == predicted:
           output_word = word
           break
   seed_text += " " + output_word  # Append the predicted word to the seed text
# Print the generated text
print(seed_text)
