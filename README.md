# Kaggle-NLP-Disaster-Tweets
This project is an exploration of possible solutions to the Kaggle NLP Disaster Tweets competition. It is scored using the f1 score, and my aim is to complete the challenge as simply as possible. 

## Attempts so far:
1. simple-sentiment v1
    - Score: 0.49249
    - This first attempt was the simplest possible solution I came up with. It consists of loading the default sentiment analysis pipeline from HuggingFace Transformers library and running the analysis with no training. 
2. simple-sentiment v2
    - Score: 0.50137
    - This second attempt made a simple change: using a better model. Instead of the default model used by HF pipeline's sentiment analysis, I used the model "cardiffnlp/twitter-roberta-base-sentiment-latest" from HuggingFace. This provided a substantial improvement on evaluation of a subset of training data, but made little difference on the submission result. 
3. trained-sentiment v1
    - Score: Not yet tested
    - This is the first attempt to train a model. Hyperparameters were chosen somewhat arbitrarily, an will be fine-tuned in a future version. 