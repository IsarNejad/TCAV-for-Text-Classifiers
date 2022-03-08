# TCAV-for-Improving-Generalizability

This repository provides the data and code related to the following publication: 

Nejadgholi, I. Fraser, K. C., Kiritchenko, S. (2022). Improving Generalizability in Implicitly Abusive Language Detection with Concept Activation Vectors. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics. 

## Data

We annoated the _Hostile_ class of the _EA-dev_ set and the _Hate_ class of the _CH_ dataset for implicit/explicit abuse. This data is used in the experiments of Section 5. Also, a subset of these datasets are used as concept examples to define the implicit and explicit concepts as described in Section 4. 

There are four files in the Data folder:

CH_Anti_Asian_hate_implicit_indexes.csv : indexes of implicitly hateful samples in the _Anti Asian Hate_ class of the _CH_ dataset. These indexes correspond to indexes of the `annotations.csv` file available at http://claws.cc.gatech.edu/covid/. 

CH_Anti_Asianhate_explicit_indexes.csv : indexes of explicitly hateful samples in the _Anti Asian Hate_ class of the _CH_ dataset. These indexes correspond to indexes of the `annotations.csv` file available at http://claws.cc.gatech.edu/covid/. 


EA_dev_hostile_implicit_ids.csv: tweet ids of implicitly hostile samples in the EA-dev set. The whole dataset is availabel at https://zenodo.org/record/3816667#.YUJPkJ1KiUk

EA_dev_hostile_explicit_ids.csv: tweet ids of explicitly hostile samples in the EA-dev set. The whole dataset is availabel at https://zenodo.org/record/3816667#.YUJPkJ1KiUk

## Software

Table4.ipynb: This colab notebook re-produces the results in Table 4. 

wiki-DoE.ipynb: This colab notebook calcuates the DoE scores used in section 6 of the paper. DoE scores are saved in 'toxic_DoEs.csv' and used by the 'augment-DoE-based.ipynb' for data augmentation. 

augment-DoE-based.ipynb: This colab notebook trains an augmented Wiki classifier discussed in section 6. The data augmnetation is based on DoE scores saved in 'toxic_DoEs.csv'. We use the trainer module from Huggingface for training a RoBerta-based binary classifier. 




Python modules:
 
word_process.py: used to preprocess tweets 

Roberta_model_data.py : class to define roberta model and to compute gradients and logits of the classifier

TCAV.py: fuctions to claculate sensitivities and the TCAV scores (Section 4) 

DoE.py: functions to calcualte the DoE score (Sections 5 and 6)
