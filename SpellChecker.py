import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


#We have three databases that we're going to train and test
filepath_dict = {'yelp':   'F:/Engineering/CSE/4th year/NLP/data/sentiment labelled sentences/yelp_labelled.txt',
                 'amazon': 'F:/Engineering/CSE/4th year/NLP/data/sentiment labelled sentences/amazon_cells_labelled.txt',
                 'imdb':   'F:/Engineering/CSE/4th year/NLP/data/sentiment labelled sentences/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)
df = pd.concat(df_list) #df now carries all sentences from the three databases; yelp, amazon, and imdb



#Function to create a matrix from the word embeddings of GloVe
#GloVe is a database with 6 Billion word embeddings
#Function parameters: filepath: path to GloVe
#                     word_index: word->index dictionary
#                     embedding_dim: dimensions of the embeddings, here we used 50 embeddings/word
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix



#Creating a model to train and test the data on
def myModel(vocab_size, embedding_dim, maxlen, embedding_matrix, X_train, labels_train, X_test, labels_test):

    model2 = Sequential()
    model2.add(layers.InputLayer(input_shape=(100,)))
    #taking advantage of the Embedding layer of keras
    #to initialize an embedding layer, three parameters are requires: input_dim which is the vocab size of the text data
    #output_dim which is the size of the output vectors for each word
    #input_length which is the length of the input sequences(sentences)
    #since i'm using GloVe, we'll use their pretrained weights and allow them to be trained to gain higher accuracy
    model2.add(layers.Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                input_length=maxlen,
                                weights=[embedding_matrix],
                                trainable=True))
    model2.add(layers.GlobalMaxPool1D())
    model2.add(layers.Dense(10, activation='relu'))
    model2.add(layers.Dense(1, activation='sigmoid'))
    model2.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
    model2.save('model.h5')
    history = model2.fit(X_train, labels_train,
                         epochs=100,
                         verbose=False,
                         validation_data=(X_test, labels_test),
                         batch_size=10)
    model2.summary()
    loss, accuracy = model2.evaluate(X_train, labels_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model2.evaluate(X_test, labels_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))



#Now, we're gonna loop over the three databases, checking their training and tests accuracies
def train_test(df):

    for source in df['source'].unique():
        df_source = df[df['source'] == source]
        sentences = df_source['sentence'].values
        labels = df_source['label'].values

        sentences_train, sentences_test, labels_train, labels_test = train_test_split(
            sentences, labels, test_size=0.25, random_state=1000)

        #Tokenizer class
        #fit_on_texts: simply creates a vocab index (word_index) based on frequency, it takes a list of sentences (sentences_train)
        #texts_to_sequences: transforms each sentence in sentences_train to a sequence of integers
        #So it basically takes each word in the text and replaces it with its corresponding integer value from the word_index
        #vocab_size is the number of vocab/words we have in word_index
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(sentences_train)
        X_train = tokenizer.texts_to_sequences(sentences_train)
        X_test = tokenizer.texts_to_sequences(sentences_test)
        vocab_size = len(tokenizer.word_index) + 1

        #maxlen is the maximum length of sentences
        #pad_sequences: pads sentences with 0 so that all sentences would be of the same length
        maxlen = 100
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        #embedding_dim is the dimension of embeddings/encodings we have for each word
        embedding_dim = 50
        embedding_matrix = create_embedding_matrix('F:/Engineering/CSE/4th year/NLP/data/glove.6B/glove.6B.300d.txt',
                                                   tokenizer.word_index, embedding_dim)
        myModel(vocab_size, embedding_dim, maxlen, embedding_matrix, X_train, labels_train, X_test, labels_test)



def preprocess_user_sentence(x, df, source):

    dataframe = df[df['source'] == source]
    sentences = dataframe['sentence'].values
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    x = tokenizer.texts_to_sequences([x])
    x = pad_sequences(x, padding='post', maxlen=100)
    return x

#train_test(df)






