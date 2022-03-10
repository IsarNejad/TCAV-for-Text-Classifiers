
## Further Requirements 
 
To use the codebase, please follow these steps:
 
1.    Train a binary roberta-based classifier with [Wiki](https://github.com/IsarNejad/cross_dataset_toxicity) dataset and save it at /models/exp-Toxic-roberta. 
2.    Using the [data annotated for explicitness](https://github.com/IsarNejad/TCAV-for-Text-Classifiers/tree/main/Data), create the following text files. Each file include the text of tweets separated by double newlines.     
 &nbsp;&nbsp;&nbsp;&nbsp; `data/EA_dev_implicit.txt`, `data/EA_dev_explicit.txt`, `data/CH_implicit.txt` and `data/CH_explicit.txt`
 
 3.    Create the text file `data/random_stopword_tweets.txt`, including the text of 2500 random tweets collected with stopwords, separated by double newlines. 

  
