# Sentiment-Analysis

### A simple sentiment analysis model: </br>
Layer one: Input layer </br>
Layer two: keras' Embedding layer, note that i used GloVe's pretrained word embeddings </br>
Layer three: Global max pooling </br>
Layer four: 10 neurons Dense layer with relu activation </br>
Layer five: 1 neuron Dense layer with sigmoid activation </br>

### Trained on labelled sentences from 'imdb' </br>
Training Accuracy: 1.0000 </br>
Testing Accuracy:  0.7680 </br>
### Trained on labelled sentences from 'amazon' </br>
Training Accuracy: 1.0000 </br>
Testing Accuracy:  0.7960 </br>
### Trained on labelled sentences from 'yelp' </br>
Training Accuracy: 1.0000 </br>
Testing Accuracy:  0.8128 </br>

#### SA.py contains the model and the function used for training
#### demo.py is used for prediction



