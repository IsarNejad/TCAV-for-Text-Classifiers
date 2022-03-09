#by Isar Nejadgholi

import numpy as np
import os
import torch.nn as nn
import torch
import transformers
import pickle


from transformers import RobertaTokenizerFast
from torch.utils.data.dataloader import DataLoader
from Roberta_model_data import RobertaClassifier,ToxicityDataset

PATH_TO_Data = ""
PATH_TO_MODELS = ""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_folder_toxic = PATH_TO_MODELS+'/models/exp-Toxic-roberta'
model_toxic = RobertaClassifier(model_folder_toxic)
tokenizer_toxic = RobertaTokenizerFast.from_pretrained(model_folder_toxic)

#model_folder_Founta = PATH_TO_MODELS+'models/exp-Founta_original_roberta'
#model_Founta = RobertaClassifier(model_folder_Founta)
#tokenizer_Founta = RobertaTokenizerFast.from_pretrained(model_folder_Founta)


#model_folder_EA = PATH_TO_MODELS+'models/exp-EA_2_class_roberta'
#model_EA = RobertaClassifier(model_folder_EA)
#tokenizer_EA = RobertaTokenizerFast.from_pretrained(model_folder_EA)



with open(PATH_TO_Data+'data/random_stopword_tweets.txt','r') as f_:
  random_examples= f_.read().split('\n\n')

with open(PATH_TO_Data+'data/EA_dev_hostile_explicit.txt','r') as f_:
  explicit_EA_dev_hate= f_.read().split('\n\n')


with open(PATH_TO_Data+'data/racism_hate_explicit.txt','r') as f_:
  explicit_racism_hate= f_.read().split('\n\n')

#explicit concept
explicit = explicit_EA_dev_hate + [explicit_racism_hate[i] for i in list(np.random.choice(len(explicit_racism_hate),86))]

def get_dataloader(X, y, tokenizer, batch_size):
  assert len(X) == len(y)
  encodings = tokenizer(X, truncation=True, padding=True, return_tensors="pt")
  dataset = ToxicityDataset(encodings, y)
  dataloader = DataLoader(dataset, batch_size=batch_size)
  return dataloader

def get_reps(model,tokenizer, concept_examples):
    #returns representations
    batch_size = 8
    concept_labels = torch.ones([len(concept_examples)]) #fake labels
        
    concept_repres = []
    concept_dataloader = get_dataloader(concept_examples,concept_labels,tokenizer,64)
    with torch.no_grad():
      for i_batch, batch in enumerate(concept_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        _, _, representation = model(input_ids, attention_mask=attention_mask)
        concept_repres.append(representation[:,0,:])
    
    concept_repres = torch.cat(concept_repres, dim=0).cpu().detach().numpy()
    #print('concept representation shape', concept_repres.shape)
    #print('concept representation shape', representation[:,0,:].shape)

    return concept_repres

def statistical_testing(model, tokenizer, concept_examples, added_example=None, num_runs=100):
  #returns CAVs
  cavs = []
  if added_example:
    concept_repres = get_reps(model,tokenizer,concept_examples+[added_example])
  else:
    concept_repres = get_reps(model,tokenizer,concept_examples)
  #print(len(concept_repres))
  for i in range(num_runs):
    #print(i)
    if added_example:  # if there is no added example it only calculates CAVs for the explicit concept
      concept_rep_ids = list(np.random.choice(range(len(concept_repres)-1), 2)) +[len(concept_repres)-1]
    else:
      concept_rep_ids = list(np.random.choice(range(len(concept_repres)), 2)) 
    #print(concept_rep_ids)
    concept_rep = [concept_repres[i] for i in concept_rep_ids]
    cavs.append(np.mean(concept_rep, axis = 0))

  return cavs

  def get_logits_grad(model, tokenizer, sample, desired_class):
    #returns logits and gradients 
    input = tokenizer(sample, truncation=True,padding=True, return_tensors="pt")
    model.zero_grad()
    input_ids = input['input_ids'].to(device)
    attention_mask = input['attention_mask'].to(device)
    logits, _, representation = model(input_ids, attention_mask=attention_mask)
    

    logits[0, desired_class].backward()
    #print('cav shape',cav.shape)
    grad = model.grad_representation
    #print('first',grad.shape)
    grad = grad[0][0].cpu().numpy()#grad.sum(dim=0).squeeze().cpu().detach().numpy()
        
    return logits,grad

def get_DoE(classifier = 'toxicity',desired_class = 1, added_example = None):
  #returns DoE if there is an added example otherwise it only calcuates TCAV for explicit concept
  if classifier=='toxicity':
    model = model_toxic
    tokenizer = tokenizer_toxic
  elif classifier=='Founta':
    model = model_Founta
    tokenizer = tokenizer_Founta
  elif classifier =='EA':
    model = model_EA
    tokenizer = tokenizer_EA
  else:
    print('model is unknown')
    return 


  examples = random_examples[:2000]
  concept_examples = explicit

  num_runs = 100
  #print('calculating cavs...')
  model.to(device)
  concept_cavs = statistical_testing(model,tokenizer, concept_examples,added_example, num_runs=num_runs)


  if os.path.exists('grads_logits/'+classifier+'_random_'+str(desired_class)+'.pkl'):
    #print('logits and grads are saved.')
    with open('grads_logits/'+classifier+'_random_'+str(desired_class)+'.pkl','rb') as handle:
      data = pickle.load(handle)
    
    grads = data['grads']
    logits = data['logits']
  else:
    #print('calculating logits and grads...')
    logits = []
    grads = []
    for sample in examples:
      logit,grad = get_logits_grad(model, tokenizer, sample, desired_class)
      grads.append(grad)
      logits.append(logit)
      data ={'grads':grads,
            'logits':logits}
    
    with open('grads_logits/'+classifier+'_random_'+str(desired_class)+'.pkl', 'wb') as handle:
      pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    


  sensitivities = [] 
  for grad in grads:
    sensitivities.append([np.dot(grad, cav) for cav in concept_cavs])
  sensitivities = np.array(sensitivities)
  tcavs = []
  
  for i in range(num_runs):
    tcavs.append(len([s for s in sensitivities[:,i] if s>0])/len(examples))
   
  #print('TCAV score for the concept: ')
  #print(np.mean(tcavs),np.std(tcavs)) 
  DoE = np.mean(tcavs)
  return DoE
  
