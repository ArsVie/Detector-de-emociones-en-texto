import numpy as np
import matplotlib as plt
import pandas as pd
import tensorflow as tf
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import warnings
import re
warnings.filterwarnings("ignore")
def clean_text(text, keep_emojis=False, fix_contractions=False):
    text = re.sub(r"\.{2,}", "...", text)  # Ellipses
    text = re.sub(r"!{2,}", "!", text)     # !
    text = re.sub(r"\?{2,}", "?", text)    # ?
    text = re.sub(r"[^a-zA-Z0-9\s!?.]", "", text.lower())

    return text
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
df=pd.read_csv('tweet_emotions.csv')
#remove sentiment=empty
df.drop(df[df['sentiment']=='empty'].index, inplace=True)
df.loc[df['sentiment'] == 'hate', 'sentiment'] = 'anger'
df.loc[df['sentiment'] == 'boredom', 'sentiment'] = 'neutral'
df.loc[df['sentiment'] == 'enthusiasm', 'sentiment'] = 'fun'
sentimentlabels=df.sentiment.unique()
text=df['content']
sentiment=df['sentiment']
print(sentiment.unique())
sentiment.replace(sentiment.unique(),range(9),inplace=True)
print(sentiment.shape,text.shape)



cleaned_text = text.apply(clean_text)
tokenized=cleaned_text.apply(word_tokenize)
tokenized=tokenized.apply(lambda x: [word for word in x if word not in stopwords.words('english')])
#padding
MAX_LEN=12
tokenizer = Tokenizer(num_words=3000, oov_token="<OOV>")
tokenizer.fit_on_texts(tokenized)
sequences = tokenizer.texts_to_sequences(tokenized)
X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
y=np.array(sentiment)
# One-hot encoding
y_onehot = to_categorical(y,num_classes=9)

#padding
MAX_LEN=12
tokenizer = Tokenizer(num_words=3000, oov_token="<OOV>")
tokenizer.fit_on_texts(tokenized)
sequences = tokenizer.texts_to_sequences(tokenized)
X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
y=np.array(sentiment)
# One-hot encoding
y_onehot = to_categorical(y,num_classes=9)
num_classes = 9

model = Sequential([
    Embedding(input_dim=3000, output_dim=128, input_length=MAX_LEN),
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

early_stopping = EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y) #para balancear el peso entre clases
class_weights = dict(enumerate(class_weights))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=1341,shuffle=True)
history = model.fit(X_train,
                    y_train,
                    epochs=50,
                    validation_data=(X_test, y_test),
                    callbacks=early_stopping,
                    class_weight=class_weights,
                    batch_size=64)
def emociones(text):
  cleaned_text=clean_text(text)
  tokenized=word_tokenize(cleaned_text)
  tokenized=[word for word in tokenized if word not in stopwords.words('english')]
  tokenized=tokenizer.texts_to_sequences([tokenized])
  tokenized=pad_sequences(tokenized, maxlen=MAX_LEN, padding="post", truncating="post")
  results=np.round(np.array(model.predict(tokenized))[0],4)
  tabla=np.array(list(zip(df.sentiment.unique(), results)))
  tabla=tabla[tabla[:,1].argsort()[::-1]]
  print(tabla[:2,0]) #Regresa las 2 emociones mas probables segun el modelo

emociones(str(input(“Ingrese un texto para analisis emocional: ”)))
input()
