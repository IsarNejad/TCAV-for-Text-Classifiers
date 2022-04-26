# TCAV for Explaining Text Classifiers 

This repository provides the data and code related to the following ACL2022 publication: 

Nejadgholi, I. Fraser, K. C., Kiritchenko, S. (2022). Improving Generalizability in Implicitly Abusive Language Detection with Concept Activation Vectors. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics. https://arxiv.org/pdf/2204.02261.pdf

## Data

As described in the paper, we annotated the _Hostile_ class of the dev set of the [East-Asian Prejudice (EA)](https://zenodo.org/record/3816667#.YUJPkJ1KiUk) dataset and the _Anti-Asian Hate_ class of the [COVID-HATE (CH)](http://claws.cc.gatech.edu/covid/) dataset for implicit/explicit abuse. Our annotations are available in the Data folder:

`CH_Anti_Asian_hate_implicit_indexes.csv` and `CH_Anti_Asianhate_explicit_indexes.csv` include indexes of implicitly and explicitly hateful samples in the _Anti-Asian Hate_ class of the _CH_ dataset, respectively. These indexes correspond to indexes of the `annotations.csv` file from the original dataset.  

`EA_dev_hostile_implicit_ids.csv` and `EA_dev_hostile_explicit_ids.csv` include tweet ids of implicitly and explicitly hostile samples of the _EA-dev_ set. 

## Software

### Python modules:
 
`Roberta_model_data.py`: Roberta model and functions to compute gradients and logits of a roberta-based classifier

`TCAV.py`: fuctions to claculate sensitivities of a trained classifier to a human-defined concept (TCAV scores described in Section 4 of the paper) 

`DoE.py`: functions to calcualte the Degree of Explicitness (DoE scores described in Sections 5 and 6 of teh paper)

### Example Notebooks:
These notebooks illusterate how to use the above functionalities. In all of the notebooks, the _Toxicity_ classifier refers to a roberta-based binary classifier trained with the [Wiki](https://github.com/IsarNejad/cross_dataset_toxicity) dataset. 

`TCAV_Example.ipynb`: This notebook shows how to calculate the sensitivity of a trained classifier to a human-defined concept (similar to the results in Table 5 of the paper. 

`DoE_example`.ipynb: This colab notebook calcuates the Degree of Explicitness (DoE scores introduced in section 5 of the paper). 
