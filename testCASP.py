import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForTokenClassification, BertTokenizerFast, EvalPrediction
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import os
import requests
from tqdm.auto import tqdm
import re

from transformers import logging as hf_logging
from sklearn.model_selection import train_test_split
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score



def downloadNetsurfpDataset():

    casp12DatasetValidUrl = 'https://www.dropbox.com/s/te0vn0t7ocdkra7/CASP12_HHblits.csv?dl=1'


    datasetFolderPath = "dataset/"

    casp12testFilePath = os.path.join(datasetFolderPath, 'CASP12_HHblits.csv')



    if not os.path.exists(datasetFolderPath):
        os.makedirs(datasetFolderPath)

    def download_file(url, filename):
        response = requests.get(url, stream=True)
        with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
            total=int(response.headers.get('content-length', 0)),
            desc=filename) as fout:
                for chunk in response.iter_content(chunk_size=4096):
                    fout.write(chunk)



    if not os.path.exists(casp12testFilePath):
        download_file(casp12DatasetValidUrl, casp12testFilePath)

         




def load_dataset(path, max_length):
    
    df = pd.read_csv(path,names=['input','dssp3','dssp8','disorder','cb513_mask'],skiprows=1)
        
    df['input_fixed'] = ["".join(seq.split()) for seq in df['input']]
    df['input_fixed'] = [re.sub(r"[UZOB]", "X", seq) for seq in df['input_fixed']]
    seqs = [ list(seq)[:max_length-2] for seq in df['input_fixed']]

    df['label_fixed'] = ["".join(label.split()) for label in df['dssp8']]
    labels = [ list(label)[:max_length-2] for label in df['label_fixed']]

    df['disorder_fixed'] = [" ".join(disorder.split()) for disorder in df['disorder']]
    disorder = [ disorder.split()[:max_length-2] for disorder in df['disorder_fixed']]

    assert len(seqs) == len(labels) == len(disorder)
    return seqs, labels, disorder


def main():
    
    
    df = pd.read_csv('protein.csv')

    data = df.loc[df['len']>= 50]


    train_seqs = [ list(seq) for seq in data['seq']]
    train_labels = [ list(label) for label in data['sst8']]
    
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model_name = 'Rostlab/prot_bert_bfd'
    max_length = 1024
    downloadNetsurfpDataset()
    
    casp12_test_seqs, casp12_test_labels, casp12_test_disorder = load_dataset('dataset/CASP12_HHblits.csv', max_length)
    
    max_length = 512
    
    output_dir='prot_bert_bfd_ss8/'
    seq_tokenizer = BertTokenizerFast.from_pretrained(output_dir, do_lower_case=False, max_length = max_length)
    
    casp12_test_seqs_encodings = seq_tokenizer(casp12_test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
   


    #"""7. Tokenize labels"""

# Consider each label as a tag for each token
    unique_tags = set(tag for doc in train_labels for tag in doc)
    unique_tags  = sorted(list(unique_tags))  # make the order of the labels unchanged
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    def encode_tags(tags, encodings):
        labels = [[tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
            doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
            encoded_labels.append(doc_enc_labels.tolist())

        return encoded_labels
        
    casp12_test_labels_encodings = encode_tags(casp12_test_labels, casp12_test_seqs_encodings)
    
    
    
    def mask_disorder(labels, masks):
        for label, mask in zip(labels,masks):
            for i, disorder in enumerate(mask):
                if disorder == "0.0":
            #shift by one because of the CLS token at index 0
                    label[i+1] = -100
                    
    mask_disorder(casp12_test_labels_encodings, casp12_test_disorder)
                    
   
    #"""9. Create SS3 Dataset"""

    class SS3Dataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
            

# we don't want to pass this to the model
    _ = casp12_test_seqs_encodings.pop("offset_mapping")

    casp12_test_dataset = SS3Dataset(casp12_test_seqs_encodings, casp12_test_labels_encodings)
    
    
    #output_dir='./results/FirstRun/checkpoint-585'

    model = AutoModelForTokenClassification.from_pretrained(output_dir,
                                                       local_files_only=True)
                                                       

    
    
    trainer = Trainer(model=model)
    trainer.model = model.cuda()
    predictions, label_ids, metrics = trainer.predict(casp12_test_dataset)

    
    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray):
        
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != torch.nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(id2tag[label_ids[i][j]])
                    preds_list[i].append(id2tag[preds[i][j]])

        return preds_list, out_label_list
    
    
    def compute_metrics(p: EvalPrediction):
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
        "accuracy": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
         "f1": f1_score(out_label_list, preds_list),
    }
    print(compute_metrics(trainer.predict(casp12_test_dataset)))
    idx = 0
    sample_ground_truth = " ".join([id2tag[int(tag)] for tag in casp12_test_dataset[idx]['labels'][casp12_test_dataset[idx]['labels'] != torch.nn.CrossEntropyLoss().ignore_index]])
    sample_predictions =  " ".join([id2tag[int(tag)] for tag in np.argmax(predictions[idx], axis=1)[np.argmax(predictions[idx], axis=1) != torch.nn.CrossEntropyLoss().ignore_index]])
    sample_sequence = seq_tokenizer.decode(list(casp12_test_dataset[idx]['input_ids']), skip_special_tokens=True)
    print("Sequence       : {} \nGround Truth is: {}\nprediction is  : {}".format(sample_sequence,
                                                                      sample_ground_truth,
                                                                      # Remove the first token on prediction becuase its CLS token
                                                                      # and only show up to the input length
                                                                      sample_predictions[2:len(sample_sequence)+2]))
    
if __name__ == "__main__":
    main()