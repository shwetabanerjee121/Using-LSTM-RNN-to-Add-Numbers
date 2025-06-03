import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Define the character set and mappings
chars = '0123456789+ '
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
num_chars = len(chars)

# Parameters
MAX_DIGITS = 3
MAXLEN = MAX_DIGITS * 2 + 1  # e.g., '345+678'
TRAINING_SIZE = 50000

def generate_data(size=TRAINING_SIZE):
    questions = []
    answers = []
    seen = set()
    while len(questions) < size:
        a = np.random.randint(1, 10**MAX_DIGITS)
        b = np.random.randint(1, 10**MAX_DIGITS)
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        q = f'{a}+{b}'
        q = q + ' ' * (MAXLEN - len(q))
        ans = str(a + b)
        ans += ' ' * (MAXLEN - len(ans))
        questions.append(q[::-1])  # Reverse the question
        answers.append(ans)
    return questions, answers

def vectorize_data(questions, answers):
    x = np.zeros((len(questions), MAXLEN, num_chars), dtype=np.bool_)
    y = np.zeros((len(answers), MAXLEN, num_chars), dtype=np.bool_)
    for i, sentence in enumerate(questions):
        for t, char in enumerate(sentence):
            x[i, t, char_to_idx[char]] = 1
    for i, sentence in enumerate(answers):
        for t, char in enumerate(sentence):
            y[i, t, char_to_idx[char]] = 1
    return x, y

# Generate and vectorize data
questions, answers = generate_data()
x, y = vectorize_data(questions, answers)

# Build the model
model = Sequential()
model.add(LSTM(128, input_shape=(MAXLEN, num_chars)))
model.add(RepeatVector(MAXLEN))
model.add(LSTM(128, return_sequences=True))
model.add(TimeDistributed(Dense(num_chars, activation='softmax')))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
es = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(x, y, batch_size=128, epochs=100, validation_split=0.2, callbacks=[es])

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Evaluate the model
def decode_sequence(seq):
    return ''.join(idx_to_char[np.argmax(vec)] for vec in seq)

for i in range(10):
    ind = np.random.randint(0, len(x))
    input_seq = x[ind:ind+1]
    decoded = model.predict(input_seq)
    question = questions[ind][::-1]
    correct = answers[ind]
    guess = decode_sequence(decoded[0])
    print(f'Q: {question} | T: {correct} | P: {guess}')
