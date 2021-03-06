import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForTokenClassification, BertTokenizerFast, EvalPrediction
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from transformers import logging as hf_logging
from sklearn.model_selection import train_test_split
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score


def main():

    df = pd.read_csv('protein.csv')

    data = df.loc[df['len']>= 50]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)



    x = data.seq.values
    y = data.sst3.values
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
    
    
    





    hf_logging.set_verbosity_info()
# retreive the saved model 
    output_dir='prot_bert_bfd_ss3/'
    seq_tokenizer = BertTokenizerFast.from_pretrained(output_dir, do_lower_case=False)
    model = AutoModelForTokenClassification.from_pretrained(output_dir,
                                                       local_files_only=True)
                                                       
                                                       
                                                       
                                                       
                                                       
    """*6*. Tokenize sequences"""
    train_seqs_encodings = seq_tokenizer(train_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
    val_seqs_encodings = seq_tokenizer(val_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
    test_seqs_encodings = seq_tokenizer(test_seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)

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

    train_labels_encodings = encode_tags(train_labels, train_seqs_encodings)
    val_labels_encodings = encode_tags(val_labels, val_seqs_encodings)
    test_labels_encodings = encode_tags(test_labels, test_seqs_encodings)


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
    _ = train_seqs_encodings.pop("offset_mapping")
    _ = val_seqs_encodings.pop("offset_mapping")
    _ = test_seqs_encodings.pop("offset_mapping")

    train_dataset = SS3Dataset(train_seqs_encodings, train_labels_encodings)
    val_dataset = SS3Dataset(val_seqs_encodings, val_labels_encodings)
    test_dataset = SS3Dataset(test_seqs_encodings, test_labels_encodings)
                                                       



    trainer = Trainer(model=model)
    trainer.model = model.cuda()
    # outputs = trainer.predict(test_dataset)
    # #print(y)
    # #logits = outputs.predictions
    # logits = outputs[0]
    # print("typw is: ", type(logits))
    # predicted_label_classes = torch.argmax(logits, dim=2)
    # #predicted_label_classes = logits.argmax(-1)
    # print(predicted_label_classes)
    # predicted_labels = [model.config.id2label[id] for id in predicted_label_classes.squeeze().tolist()]
    # print(predicted_labels)
    
    predictions, label_ids, metrics = trainer.predict(test_dataset)
    print(metrics)
    
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
    print(compute_metrics(trainer.predict(test_dataset)))
    idx = 200
    sample_ground_truth = " ".join([id2tag[int(tag)] for tag in test_dataset[idx]['labels'][test_dataset[idx]['labels'] != torch.nn.CrossEntropyLoss().ignore_index]])
    sample_predictions =  " ".join([id2tag[int(tag)] for tag in np.argmax(predictions[idx], axis=1)[np.argmax(predictions[idx], axis=1) != torch.nn.CrossEntropyLoss().ignore_index]])
    sample_sequence = seq_tokenizer.decode(list(test_dataset[idx]['input_ids']), skip_special_tokens=True)
    print("Sequence       : {} \nGround Truth is: {}\nprediction is  : {}".format(sample_sequence,
                                                                      sample_ground_truth,
                                                                      # Remove the first token on prediction becuase its CLS token
                                                                      # and only show up to the input length
                                                                      sample_predictions[2:len(sample_sequence)+2]))
    
if __name__ == "__main__":
    main()