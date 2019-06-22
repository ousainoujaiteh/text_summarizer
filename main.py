from __future__ import print_function

import pandas as pd
from library.seq2seq import Seq2SeqSummarizer
import numpy as np

np.random.seed(42)
data_dir_path = './runner/data' # refers to the demo/data folder
model_dir_path = './runner/models' # refers to the demo/models folder

print('loading csv file ...')
df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")
X = df['text']
Y = df.title

config = np.load(Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path),allow_pickle=True).item()

summarizer = Seq2SeqSummarizer(config)
summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

print('start predicting ...')
for i in range(60,80):
    x = X[i]
    actual_headline = Y[i]
    headline = summarizer.summarize(x)
    print("\n===============================================================================================================")
    print("\n===============================================================================================================")
    print('Article: ', x)
    print("Output***********************************************************************************************************")
    print('Generated Headline: ', headline)
    print('Original Headline: ', actual_headline)

#79