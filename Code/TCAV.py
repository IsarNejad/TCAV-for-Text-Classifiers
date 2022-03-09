#by Isar Nejadgholi
import torch.nn as nn
import numpy as np
import os
import pickle
import torch
from transformers import RobertaTokenizerFast
from torch.utils.data.dataloader import DataLoader
from Roberta_model_data import RobertaClassifier,ToxicityDataset
import random

random.seed(100)


PATH_TO_Data = 'PATH_TO_Data'
PATH_TO_Model = 'PATH_TO_Model'

with open(PATH_TO_Data+'data/random_stopword_tweets.txt','r') as f_:
  random_examples= f_.read().split('\n\n')

random_concepts = random_examples[-100:]


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model_folder_toxic = PATH_TO_Model+'exp-Toxic-roberta'
model_folder_Founta = PATH_TO_Model+'exp-Founta_original_roberta'
model_folder_EA = PATH_TO_Model+'exp-EA_2_class_roberta'
model_folder_CH = PATH_TO_Model+'CH_roberta'
model_folder_CH_explicit = PATH_TO_Model+'explicit_CH_roberta'
model_folder_toxic_explicit = PATH_TO_Model+'explicit_wiki_roberta'


def get_dataloader(X, y, tokenizer, batch_size):
  assert len(X) == len(y)
  encodings = tokenizer(X, truncation=True, padding=True, return_tensors="pt")
  dataset = ToxicityDataset(encodings, y)
  dataloader = DataLoader(dataset, batch_size=batch_size)
  return dataloader

def get_reps(model,tokenizer, concept_examples):
  #returns roberta representations    
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

def statistical_testing(model, tokenizer, concept_examples, num_runs=10):
  #calculates CAVs
  cavs = []

  concept_repres = get_reps(model,tokenizer,concept_examples)
  for i in range(num_runs):
    #print(i)
    concept_rep_ids = list(np.random.choice(range(len(concept_repres)), 5))
    concept_rep = [concept_repres[i] for i in concept_rep_ids]
    cavs.append(np.mean(concept_rep, axis = 0))

  return cavs

def get_logits_grad(model, tokenizer, sample, desired_class):
  #returns logits and gradients
  #print(sample)
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

def get_preds_tcavs(classifier = 'toxicity',desired_class = 1,examples_set = 'random',concept_examples = random_concepts, num_runs = 10):
  #returns logits, sensitivies and tcav score
  if classifier=='toxicity':
    model = RobertaClassifier(model_folder_toxic)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_folder_toxic)
  elif classifier=='Founta':
    model = RobertaClassifier(model_folder_Founta)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_folder_Founta)
  elif classifier =='EA':
    model = RobertaClassifier(model_folder_EA)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_folder_EA)
  elif classifier =='CH':
    model = RobertaClassifier(model_folder_CH)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_folder_CH)
  elif classifier =='CH_exp':
    model = RobertaClassifier(model_folder_CH_explicit)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_folder_CH_explicit)  
  elif classifier =='wiki_exp':  
    model = RobertaClassifier(model_folder_toxic_explicit)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_folder_toxic_explicit)
  else:
    print('model is unknown')
    return 


  if examples_set=='random':
    examples = random_examples[:2000]#random_examples
  else:
    print('examples are unknown')
    return


  print('calculating cavs...')
  model.to(device)
  concept_cavs = statistical_testing(model,tokenizer, concept_examples, num_runs=num_runs)


  if os.path.exists('grads_logits/'+classifier+'_'+examples_set+'_'+str(desired_class)+'.pkl'):
    print('logits and grads are saved.')
    with open('grads_logits/'+classifier+'_'+examples_set+'_'+str(desired_class)+'.pkl','rb') as handle:
      data = pickle.load(handle)
    
    grads = data['grads']
    logits = data['logits']
  else:
    print('calculating logits and grads...')
    logits = []
    grads = []
    for sample in examples:
      logit,grad = get_logits_grad(model, tokenizer, sample, desired_class)
      grads.append(grad)
      logits.append(logit)
      data ={'grads':grads,
            'logits':logits}
    
    with open('grads_logits/'+classifier+'_'+examples_set+'_'+str(desired_class)+'.pkl', 'wb') as handle:
      pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
   
  sensitivities = [] 
  for grad in grads:
    sensitivities.append([np.dot(grad, cav) for cav in concept_cavs])
  sensitivities = np.array(sensitivities)
  tcavs = []
  for i in range(num_runs):
    tcavs.append(len([s for s in sensitivities[:,i] if s>0])/len(examples))
   
  print('TCAV score for the concept: ')
  print(np.mean(tcavs),np.std(tcavs)) 
  
  return logits, sensitivities, tcavs
