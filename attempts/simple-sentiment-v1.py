# Installing transformers library is necessary in Kaggle notebooks
# !pip install -q transformers

# Imports
import pandas as pd
from transformers import pipeline

# Load the files
test = pd.read_csv('../test.csv')
sample = pd.read_csv('../sample_submission.csv')

# Create the sentiment analysis pipeline and make predictions
sentiment_pipeline = pipeline("sentiment-analysis")
predictions = sentiment_pipeline(list(test['text']))
labels = [1*(i['label'] == 'NEGATIVE') for i in predictions]

# Write predictions to submission file
sample['target'] = labels
sample[['id', 'target']].to_csv('submission.csv', index=False)