
import torch.nn as nn
import numpy as np 
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
import torch


class ToxicityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class RobertaClassifier(nn.Module):
  def __init__(self, folder):
    super(RobertaClassifier, self).__init__()
    self.roberta_classifier = RobertaForSequenceClassification.from_pretrained(folder)
  
    self.grad_representation = None
    self.representation = None
    # using the representation of layer12 in the transformer
    for name, module in self.roberta_classifier.named_modules():
      #print(name)
      if name =="roberta.encoder.layer.11.output": #roberta representation 
        #print(name)
        module.register_forward_hook(self.forward_hook_fn)
        module.register_backward_hook(self.backward_hook_fn)

    self.roberta_classifier.requires_grad_(True)
  
  def forward_hook_fn(self, module, input, output):  #gradient
    self.representation = output

  def backward_hook_fn(self, module, grad_input, grad_output):
    self.grad_representation = grad_output[0]

  def forward(self, input_ids: torch.Tensor, attention_mask:torch.tensor, labels=None):
    if labels is not None:
      loss, logits = self.roberta_classifier(input_ids=input_ids, attention_mask= attention_mask, labels=labels)
    else:
      out = self.roberta_classifier(input_ids=input_ids, attention_mask=attention_mask)
      logits = out[0]

    preds = torch.argmax(logits, dim=-1)  # (batch_size, )

    if labels is None:
      return logits, preds, self.representation
    else:
      loss = nn.functional.cross_entropy(logits, labels)
      return loss, logits, preds, self.representation

  def forward_from_representation(self, representation: torch.Tensor):
    #classifier = nn.Sequential(self.xlnet_classifier.sequence_summary, self.xlnet_classifier.logits_proj)
    logits = self.roberta_classifier.classifier(representation)  # (batch_size, num_labels)
    
    preds = torch.argmax(logits, dim=-1)  # (batch_size, )
    return logits,preds