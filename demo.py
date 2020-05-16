import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from SpellChecker import preprocess_user_sentence

filepath_dict = {'yelp':   'F:/Engineering/CSE/4th year/NLP/data/sentiment labelled sentences/yelp_labelled.txt',
                 'amazon': 'F:/Engineering/CSE/4th year/NLP/data/sentiment labelled sentences/amazon_cells_labelled.txt',
                 'imdb':   'F:/Engineering/CSE/4th year/NLP/data/sentiment labelled sentences/imdb_labelled.txt'}
df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)
df = pd.concat(df_list) #df now carries all sentences from the three databases; yelp, amazon, and imdb


x = preprocess_user_sentence("loved it", df, 'yelp')
model = load_model('model.h5')


if(model.predict_classes(x) == 0):
    print("-ve")
else:
    print("+ve")