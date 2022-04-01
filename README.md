# Twitter-Sentiment-Analysis

A simple sentiment analysis web application with Malay tweets.

![Application preview](https://github.com/Quart3z/Twitter-Sentiment-Analysis/blob/master/screenshot.png)

## Models
### Word2Vec
A Word2Vec algorithm is a technique used prominently in Natural Language Processing. As a shallow network, it only consists of 2 hidden layers and an output layer: 
- Input layer, encode each of the received words into a one-hot vector.
- Hidden layer, responsible for the convergence of the model based on the features of the word.
- Output layer, produces a set of vector presentations for each of the words in the vocabulary.

By measuring the cosine similarity between the word vectors, it is possible to find the similarity score, the more similar the words are, the closer the score to 1. 

Since the result of sentiment analysis is highly based on the vocabulary of the model, large and robust vocabulary of the dataset is highly preferred for the training. Though, for a better F1 score of the classification model, vocabulary learnt should not be biased toward certain sentiment alignment. 

It is possible to improve the performance of the model by simply removing irrelevant words from the dictionary. Removing stopwords from the dataset or uncommon words helps to reduce the noise and irrelevant features from the dataset, thus reducing the overfitting of the result. On the other hand, it is not an easy task to determine the words to be removed, removing the wrong word changes its context and the sentiment of the sentence, affecting the performance of the model.

### CNN Classification
The model classifies the sentences into 2 categories: positive label and negative label. It takes a string of words as an input and embedded the sentence into stacks of vectors with the Word2Vec model trained earlier. 

The network consists of:
- 3 convolutional layers, each with a different filter size and responsible for the feature extraction from the word vectors. 
- A pooling layer with max pooling method for the down sampling of the features.
- Output layer, with softmax activation function for the determination of the sentiment score of the sentence.

A general strategy for getting a good F1-score for the model is trained with a balanced dataset. Another relevant strategy includes performing over-sample on minor classes while under-sample the major classes, allowing the model to emphasize more on the minor class instead of the major class.

One of the trade-offs of the model includes using a lower learning rate for the training process. While lower learning rate helps in finding the global minima during gradient descent, it also significantly increases the training time as the model takes longer time to converge. On the other hand, using a higher learning rate reduces the time required for the training, but at the same time, it may escape the global minima and obtain a worse result.

---

## F1-score as the evalutaion metric
Since F1-score is calculated based on the mean of accuracy and recall of the model, it takes false predictions into account, giving an evaluation on overall predictive performance of the model. As a result, F1-score is more credible in evaluating training models with an imbalanced dataset as it is not biased toward a specific class.

Other relevant metrics such as accuracy are suitable to evaluate the performance of the model. While it measures the rate of correct predictions, its score may not be credible if the dataset is imbalanced. 
