import torch


from transformers import BertModel, BertTokenizer


import re
import gc
import os
import pandas as pd
import requests
from tqdm.auto import tqdm
from torch import cuda
import numpy as np





def main():
    
    
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)

    
    df = pd.read_csv('protein.csv')
    data = df.loc[df['len']>= 50]
    data.to_csv('proteinFinal.csv')
    
    from sklearn.model_selection import train_test_split
    train_seqs = [ list(seq) for seq in data['seq']]
    train_labels = [ list(label) for label in data['sst8']]

    x = data.seq.values
    y = data.sst8.values
    # set aside 20% of train and test data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(x, y,
        test_size=0.2, shuffle = True, random_state = 8)

    # Use the same function above for the validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
    test_size=0.1,shuffle = True, random_state= 8) # 0.25 x 0.8 = 0.2
    

    train_seqs = [ list(seq) for seq in X_train]
    train_labels = [ list(label) for label in y_train]
    
    val_seqs = [ list(seq) for seq in X_val]
    val_labels = [ list(label) for label in y_val]

    test_seqs = [ list(seq) for seq in X_test]
    test_labels = [ list(label) for label in y_test]
    
    
        
    model_name = 'Rostlab/prot_bert_bfd'
   
    seq_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False )
    model = BertModel.from_pretrained(model_name)
    model = model.to(device)
    model = model.eval()
    

    def embed_dataset(dataset_seqs, shift_left = 0, shift_right = -1):
        inputs_embedding = []

        for sample in tqdm(dataset_seqs):
            with torch.no_grad():
                ids = seq_tokenizer.batch_encode_plus([sample], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
                embedding = model(input_ids=ids['input_ids'].to(device))[0]
                inputs_embedding.append(embedding[0].detach().cpu().numpy()[shift_left:shift_right])

        return inputs_embedding
    
    shift_left = 1
    shift_right = -1
    
    train_seqs_embd = embed_dataset(train_seqs, shift_left, shift_right)
    test_seqs_embd = embed_dataset(test_seqs, shift_left, shift_right)
    
    X_train = train_seqs_embd[0]
    for i in range(1,len(train_seqs)):
  #print(X_train.shape)
        X_train = np.concatenate( (X_train, np.array(train_seqs_embd[i])), axis =0)
    
    print(X_train.shape)
    y_train = train_labels[:][0]
    
    for i in range(1,len(train_seqs)):
        y_train = np.concatenate( (y_train, np.array(train_labels[:][i])), axis =0)
        
    print(y_train.shape)
        
    X_test = test_seqs_embd[0]
    for i in range(1,len(test_seqs)):
  #print(X_train.shape)
        X_test = np.concatenate( (X_test, np.array(test_seqs_embd[i])), axis =0)
        
    y_test = test_labels[:][0]
    for i in range(1, len(test_seqs)):
  #print(X_train.shape)
        y_test = np.concatenate( (y_test, np.array(test_labels[:][i])), axis =0)
        
    y_test = y_test.reshape(-1)
        
    from sklearn.linear_model import LogisticRegression

    lr_clf = LogisticRegression(max_iter=3000)
    lr_clf.fit(X_train, y_train)
    print(lr_clf.score(X_test, y_test))

if __name__ == "__main__":
    main()     
    