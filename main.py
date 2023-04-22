import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Read in the text file and convert it into a list of strings
with open('text.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# Ask the user for the number of lines of poetry to generate
num_lines = int(input("Enter the number of lines of poetry to generate: "))

# Set the default number of words per line
num_words_per_line = 4

# Preprocess the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Define the Keras model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=max_sequence_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, np.ones(len(lines)), epochs=50, batch_size=32)

# Generate poetry
for i in range(num_lines):
    input_text = np.random.choice(lines)
    input_seq = tokenizer.texts_to_sequences([input_text])[0]
    generated_seq = model.predict(pad_sequences([input_seq], maxlen=max_sequence_length, padding='post'))
    generated_text = tokenizer.sequences_to_texts([[np.argmax(generated_seq) + 1]])[0]
    print(f"Line {i+1}: {generated_text}")

