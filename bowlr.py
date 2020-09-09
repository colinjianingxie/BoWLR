

import torch
import torch.utils.data as tud
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter, defaultdict
import operator
import os, math
import numpy as np
import random
import copy
# from nltk import word_tokenize

def word_tokenize(s):
    return s.split()

# set the random seeds so the experiments can be replicated exactly
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if torch.cuda.is_available():
    torch.cuda.manual_seed(53113)

# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

def load_data(data_file):
    data = []
    with open(data_file,'r', encoding= "Latin-1") as fin:
        for line in fin:
            label, content = line.split(",", 1)
            data.append((content.lower(), label))
    return data
data_dir = "large_movie_review_dataset"
train_data = load_data(os.path.join(data_dir, "train.txt"))
dev_data = load_data(os.path.join(data_dir, "dev.txt"))

def load_test_data(data_file):
    data = []
    with open(data_file,'r', encoding= "Latin-1") as fin:
        for line in fin:
            data.append(line.strip())
    return data

test_data = load_test_data(os.path.join(data_dir, "test.txt"))

print("number of TRAIN data", len(train_data))
print("number of DEV data", len(dev_data))

"""We define a generic model class as below. The 

*   List item
*   List item

model has 2 functions, train and classify.
"""

VOCAB_SIZE = 5000
class Model:
    def __init__(self, data):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = Counter([word for content, label in data for word in word_tokenize(content)]).most_common(VOCAB_SIZE-1) 
        self.word_to_idx = {k[0]: v+1 for v, k in enumerate(self.vocab)} # word to index mapping
        self.word_to_idx["UNK"] = 0 # all the unknown words will be mapped to index 0
        self.idx_to_word = {v:k for k, v in self.word_to_idx.items()}
        self.label_to_idx = {POS_LABEL: 0, NEG_LABEL: 1}
        self.idx_to_label = [POS_LABEL, NEG_LABEL]
        self.vocab = set(self.word_to_idx.keys())
        
    def train_model(self, data):
        '''
        Train the model with the provided training data
        '''
        raise NotImplementedError
        
    def classify(self, data):
        '''
        classify the documents with the model
        '''
        raise NotImplementedError

"""## Sentiment Analysis with Logistic Regression and Bag of Words

You will implement logistic regression with bag of words features in the following.
"""

class TextClassificationDataset(tud.Dataset):
    '''
    PyTorch provide a common dataset interface. 
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    The dataset encodes documents into indices. 
    With the PyTorch dataloader, you can easily get batched data for training and evaluation. 
    '''
    def __init__(self, word_to_idx, data):
        
        self.data = data
        self.word_to_idx = word_to_idx
        self.label_to_idx = {POS_LABEL: 0, NEG_LABEL: 1}
        self.vocab_size = VOCAB_SIZE
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = np.zeros(self.vocab_size)
        
        item = torch.from_numpy(item)
        if len(self.data[idx]) == 2: # in training or evaluation, we have both the document and label
            for word in word_tokenize(self.data[idx][0]):
                item[self.word_to_idx.get(word, 0)] += 1
            label = self.label_to_idx[self.data[idx][1]]
            return item, label
        else: # in testing, we only have the document without label
            for word in word_tokenize(self.data[idx]):
                item[self.word_to_idx.get(word, 0)] += 1
            return item

best_model = None
class BoWLRClassifier(nn.Module, Model):
    '''
    Define your logistic regression model with bag of words features.
    '''
    def __init__(self, data):
        nn.Module.__init__(self)
        Model.__init__(self, data)
        
        # TODO
        self.linear = nn.Linear(VOCAB_SIZE, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        # pass
        
        
    def forward(self, bow):
        '''
        Run the model. You may only need to run the linear layer defined in the init function. 
        '''
        out = self.linear(bow)
        # TODO
        return out
    
    def train_epoch(self, train_data):

        dataset = TextClassificationDataset(self.word_to_idx, train_data)
        dataloader = tud.DataLoader(dataset, batch_size=8, shuffle=True)
        self.train()
        for i, (X, y) in enumerate(dataloader):
            X = X.float()
            y = y.long()
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            self.optimizer.zero_grad()
            preds = self.forward(X)
            loss = self.loss_fn(preds, y)
            loss.backward()
            if i % 500 == 0:
                print("loss: {}".format(loss.item()))
            self.optimizer.step()
    
    def train_model(self, train_data, dev_data):

        dev_accs = [0.]
        for epoch in range(3):
            self.train_epoch(train_data)
            dev_acc = self.evaluate(dev_data)
            print("dev acc: {}".format(dev_acc))
            if dev_acc > max(dev_accs):
                best_model = copy.deepcopy(self)
            dev_accs.append(dev_acc)

    def classify(self, docs):
        '''
        This function classifies documents into their categories. 
        docs are documents only, without labels.
        '''
        dataset = TextClassificationDataset(self.word_to_idx, docs)
        dataloader = tud.DataLoader(dataset, batch_size=1, shuffle=False)
        results = []
        with torch.no_grad():
            for i, X in enumerate(dataloader):
                X = X.float()
                if torch.cuda.is_available():
                    X = X.cuda()
                preds = self.forward(X)
                results.append(preds.max(1)[1].cpu().numpy().reshape(-1))
        results = np.concatenate(results)
        results = [self.idx_to_label[p] for p in results]
        return results
                
    def evaluate(self, data):
        '''
        This function evaluate the data with the current model. 
        data contains documents and labels. 
        It calls function "classify" to make predictions, 
        and compare with the correct labels to return the model accuracy on "data". 
        '''
        self.eval()
        preds = self.classify([d[0] for d in data])
        targets = [d[1] for d in data]
        correct = 0.
        total = 0.
        for p, t in zip(preds, targets):
            if p == t: 
                correct += 1
            total += 1
        return correct/total

lr_model = BoWLRClassifier(train_data)
if torch.cuda.is_available():
    lr_model = lr_model.cuda()
lr_model.train_model(train_data, dev_data)


preds = lr_model.classify(test_data)
def write_to_file(preds, filename):
    i = 0
    with open(os.path.join(filename), "w") as fout:
        fout.write("index,label\n")
        for pred in preds:
            fout.write("{},{}\n".format(i, pred))
            i += 1

write_to_file(preds, "lr_test_preds.txt")

weights = lr_model.linear.weight
pos_weights = weights[0]
neg_weights = weights[1]

mpv1, max_pos_weights_ind = torch.topk(pos_weights, 10, largest=True)
mnv1, max_neg_weights_ind = torch.topk(neg_weights, 10, largest=True)

mpv2, min_pos_weights_ind = torch.topk(pos_weights, 10, largest=False)
mnv2, min_neg_weights_ind = torch.topk(neg_weights, 10, largest=False)




dataset = lr_model.word_to_idx


max_pos_weights = []
max_neg_weights = []
min_pos_weights = []
min_neg_weights = []

for value, index in dataset.items():
	if index in max_pos_weights_ind:
		max_pos_weights.append(value)
	if index in max_neg_weights_ind:
		max_neg_weights.append(value)
	if index in min_pos_weights_ind:
		min_pos_weights.append(value)
	if index in min_neg_weights_ind:
		min_neg_weights.append(value)

"""

Top 10 features with the maximum weights for POSITIVE are: ['very', 'will', 'also', 'great', 'well', 'best', 'love', 'still', 'both', 'wonderful']

Top 10 features with the maximum negative weights for POSITIVE category are: ['just', 'even', 'only', 'no', 'any', 'bad', 'acting', 'nothing', 'worst', 'waste']. 

Top 10 features with the maximum positive weights for NEGATIVE category are:  ['just', 'even', 'only', 'no', 'any', 'bad', 'plot', 'nothing', 'worst', 'waste']. I noticed that "worst", "waste", "nothing", "bad", "no", "only", "just", "even" all appear here, which are very similar to the Maximum-negative weights in the POSITIVE category.

Top 10 features with the maximum negative weights for NEGATIVE category are: ['as', 'you', 'very', 'also', 'great', 'well', 'best', 'love', 'still', 'excellent']. It seems like there is an overlap in words with the maximum POSITIVE weights, which makes sense.
"""