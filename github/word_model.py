import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import os


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

folder_path = 'word_model_data'
file_list = glob.glob(os.path.join(folder_path, '*.xlsx'))
all_feats = []
all_targets = []

max_series_days = 40

for filepath in file_list:
    data = pd.read_excel(filepath)
    feats = data.iloc[:, :10]
    targets = data.iloc[:, 10]

    deq = deque(maxlen=max_series_days)
    for i in range(len(feats)):
        deq.append(list(feats.iloc[i]))
        if len(deq) == max_series_days:
            seq_to_normalize = np.array(deq)
            scaler = StandardScaler()
            scaler.fit(seq_to_normalize)
            normalized_seq = scaler.transform(seq_to_normalize)
            all_feats.append(list(normalized_seq))
            all_targets.append(targets.iloc[i])


x = np.array(all_feats)
y = to_categorical(all_targets, num_classes=18)

data = list(zip(x, y))
random.shuffle(data)
x, y = zip(*data)
x = np.array(x)
y = np.array(y)

total_num = len(x)
train_num = int(total_num * 0.8)
val_num = int(total_num * 0.9)

X_train, Y_train = x[:train_num], y[:train_num]
x_val, y_val = x[train_num:val_num], y[train_num:val_num]
X_test, Y_test = x[val_num:], y[val_num:]

def create_model():
    model = keras.models.Sequential([
        keras.layers.Conv1D(filters=16, kernel_size=3, padding='same', activation='relu',
                            input_shape=(max_series_days, 10)),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(18, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.summary()
history = model.fit(X_train, Y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val))
model.save('model1.keras')
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2,2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {test_acc:.3f}, Test loss: {test_loss:.3f}')
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(Y_test, axis=1)
labels = ["I", "You", "He", "Like", "Study", "This", "What", "Where", "Book", "Go",
          "Eat", "How", "Together", "Want", "Name", "Pen", "Good", "Bad"]
conf_matrix = confusion_matrix(y_true, y_pred_classes)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
conf_matrix_str = np.array([["{:.2f}".format(value) for value in row] for row in conf_matrix_normalized])
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized, annot=conf_matrix_str, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

