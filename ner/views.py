from django.shortcuts import render
from django.http import HttpResponse

import pandas as pd
import numpy as np
import json
np.random.seed(0)
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras

def home(request):
  
    if request.method== 'POST':

        if 'test-data' in request.POST:

            words = []
            tags = []
            nums = []
            sentences = []

            f1 = "words.pkl"
            f2 = "tags.pkl"
            f3 = "nums.pkl"
            f4 = "sentences.pkl"

            of1 = open(f1, "rb")
            of2 = open(f2, "rb")
            of3 = open(f3, "rb")
            of4 = open(f4, "rb")

            words = pickle.load(of1)
            tags = pickle.load(of2)
            nums = pickle.load(of3)
            sentences = pickle.load(of4)

            of1.close()
            of2.close()
            of3.close()
            of4.close()
            

            word2idx = {w: i + 1 for i, w in enumerate(words)}
            tag2idx = {t: i for i, t in enumerate(tags)}

            max_len = 50

            X = [[word2idx[w[0]] for w in s] for s in sentences]
            X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=nums[0]-1)

            y = [[tag2idx[w[2]] for w in s] for s in sentences]
            y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            
            model = keras.models.load_model('our_model')
            
            i = np.random.randint(0, x_test.shape[0])
            p = model.predict(np.array([x_test[i]]))
            p = np.argmax(p, axis=-1)
            y_true = y_test[i]
        
            sent = []
            labs = []

            for w, true, pred in zip(x_test[i], y_true, p[0]):
                sent.append(words[w-1])
                labs.append(tags[true])
            

            sen = " ".join(sent[0:sent.index('.')])
            ent = []
            for i in range(0,sent.index('.')):
                if(labs[i]=='O'):
                    pass
                else:
                    ent.append({"word":sent[i],"tag":labs[i]})     
                
            return render(request,'ner/home.html',{"sentence":sen,"entities":ent})

        elif 'load-data' in request.POST:
            
            df = pd.read_csv("ner_dataset.csv", encoding="latin1")
            df = df.fillna(method="ffill")
            df = df.rename(columns={'Sentence #': 'Sentence'})
            json_records = df.head(10).reset_index().to_json(orient ='records')
            data = []
            data = json.loads(json_records)
            context = {"isDataLoaded":True,'d': data}
    
            return render(request,'ner/home.html',context)
        
        elif 'process-data' in request.POST:

            df = pd.read_csv("ner_dataset.csv", encoding="latin1")
            df = df.fillna(method="ffill")
            
            uniw = df['Word'].nunique()
            unit = df['Tag'].nunique()

            words = list(set(df["Word"].values))
            words.append("ENDPAD")
            num_words = len(words)

            tags = list(set(df["Tag"].values))
            num_tags = len(tags)

            nums = [num_words,num_tags]

            class SentenceGetter(object):
                def __init__(self, data):
                    self.n_sent = 1
                    self.data = data
                    self.empty = False
                    agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                                    s["POS"].values.tolist(),
                                                                    s["Tag"].values.tolist())]
                    self.grouped = self.data.groupby("Sentence #").apply(agg_func)
                    self.sentences = [s for s in self.grouped]
                
                def get_next(self):
                    try:
                        s = self.grouped["Sentence: {}".format(self.n_sent)]
                        self.n_sent += 1
                        return s
                    except:
                        return None
            
            getter = SentenceGetter(df)
            sentences = getter.sentences
            
            f1 = "words.pkl"
            f2 = "tags.pkl"
            f3 = "nums.pkl"
            f4 = "sentences.pkl"

            of1 = open(f1, "wb")
            of2 = open(f2, "wb")
            of3 = open(f3, "wb")
            of4 = open(f4, "wb")

            pickle.dump(words, of1)
            pickle.dump(tags, of2)
            pickle.dump(nums, of3)
            pickle.dump(sentences, of4)

            of1.close()
            of2.close()
            of3.close()
            of4.close()

            df = df.rename(columns={'Sentence #': 'Sentence'})
            json_records = df.head(10).reset_index().to_json(orient ='records')
            data = []
            data = json.loads(json_records)
            
            context ={
                "isDataProcessed":True,
                "isDataLoaded":True,
                "d": data,
                "uniw":uniw,
                "unit":unit,
                "twords":num_words,
                "ttags":num_tags,
                }

            return render(request,'ner/home.html',context)

        elif 'train-model' in request.POST:

            words = []
            tags = []
            nums = []
            sentences = []

            f1 = "words.pkl"
            f2 = "tags.pkl"
            f3 = "nums.pkl"
            f4 = "sentences.pkl"

            of1 = open(f1, "rb")
            of2 = open(f2, "rb")
            of3 = open(f3, "rb")
            of4 = open(f4, "rb")

            words = pickle.load(of1)
            tags = pickle.load(of2)
            nums = pickle.load(of3)
            sentences = pickle.load(of4)

            of1.close()
            of2.close()
            of3.close()
            of4.close()
            

            word2idx = {w: i + 1 for i, w in enumerate(words)}
            tag2idx = {t: i for i, t in enumerate(tags)}

            max_len = 50

            X = [[word2idx[w[0]] for w in s] for s in sentences]
            X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=nums[0]-1)

            y = [[tag2idx[w[2]] for w in s] for s in sentences]
            y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

            input_word = Input(shape=(max_len,))
            model = Embedding(input_dim=nums[0], output_dim=50, input_length=max_len)(input_word)
            model = SpatialDropout1D(0.1)(model)
            model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
            out = TimeDistributed(Dense(nums[1], activation="softmax"))(model)
            model = Model(input_word, out)
            model.summary()

            model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

            chkpt = ModelCheckpoint("model_weights.h5", monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=True, mode='min')

            early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, verbose=0, mode='max', baseline=None, restore_best_weights=False)

            callbacks = [chkpt, early_stopping]

            history = model.fit(
                x=x_train,
                y=y_train,
                validation_data=(x_test,y_test),
                batch_size=32,
                epochs=3,
                callbacks=callbacks,
                verbose=1
            )

            model.save("our_model")
            model.evaluate(x_test, y_test)

            df = pd.read_csv("ner_dataset.csv", encoding="latin1")
            df = df.fillna(method="ffill")
            
            uniw = df['Word'].nunique()
            unit = df['Tag'].nunique()

            df = df.rename(columns={'Sentence #': 'Sentence'})
            json_records = df.head(10).reset_index().to_json(orient ='records')
            data = []
            data = json.loads(json_records)

            context ={
                "isDataProcessed":True,
                "isDataLoaded":True,
                "isModelTrained":True,
                "uniw":uniw,
                "unit":unit,
                "twords":nums[0],
                "ttags":nums[1],
                "d":data,
            }

            return render(request,'ner/home.html',context)

    return render(request,'ner/home.html')
