from torchvision import transforms
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import DataLoader, Dataset
from PIL import Image 
from  _config import Config as config
import numpy as np
import torch.nn as nn
import pickle
import _utils

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>" , 2:"<EOS>" , 3:"<UNK>", 4:"?"} #itos = index to string
        self.stoi = {"<PAD>":0 ,"<SOS>":1 ,"<EOS>":2 ,"<UNK>":3,"?":4 }   #stoi = string to index
        self.freq_threshold = freq_threshold
        self.en_model = spacy.load('en_core_web_sm')

    def __len__(self):
        return len(self.itos)

    
    def tokenizer_eng(self, sentence):
        a = [token.text for token in self.en_model.tokenizer(sentence)]
        return a


    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 5

        for sentence in sentence_list:
            for word in sentence.split():
                if word[-1]=="?":
                    word = word[0:-1]
            if word not in frequencies:
                frequencies[word]=1
            else:
                frequencies[word]+=1
            if frequencies[word] == self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx+=1


    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        liss = []
        for token in tokenized_text:
          if token in list(self.stoi.keys()):
            liss.append(self.stoi[token])
          else:
            liss.append(self.stoi["<UNK>"])
        return liss
  

class VQADataset(Dataset):
  
    def __init__(self, vqa_path, images_path, transforms=None, freq_threshold=1, train_vocab=None):
        self.train_vocab = train_vocab
        self.path = vqa_path
        self.image_path = images_path
        self.freq_threshold = freq_threshold
        self.transforms = transforms

        # Get questions and answers
        questions = []
        answers = []
        images = []
        with open(self.path,"rb") as f:
            x = pickle.load(f)

        for i in range(len(x)):
            tmp = list(x[i].keys())[0]
            answers.append(x[i][tmp]['ans'])
            questions.append(x[i][tmp]['ques'])
            images.append(list(x[i].keys())[0])

        self.image_ids = images
        self.questions = questions
        self.answers = answers


    # Initialize vocab and build it
        self.vocab = Vocabulary(self.freq_threshold)
        if not self.train_vocab:
            self.vocab.build_vocabulary(questions)
        else:
            self.vocab.stoi = train_vocab[0]
            self.vocab.itos = train_vocab[1]


    def __len__(self):
        return len(self.questions)


    def __getitem__(self, index):
        img_id = self.image_ids[index]
        image_path = self.image_path +"/"+img_id+".jpg"
        
        image = Image.open(image_path)
        image = np.array(image)
        image = torch.Tensor(image)

        if self.transforms is not None:
            image = self.transforms(image)

        normalised_question = [self.vocab.stoi["<SOS>"]]
        normalised_question += self.vocab.numericalize(self.questions[index])
        normalised_question.append(self.vocab.stoi["<EOS>"])
        ans = self.answers[index]
        return {'img':image, 'question':torch.Tensor(normalised_question), 'answer':ans}


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx


    def __call__(self, batch):
        imgs=[]
        for i in batch:
            imgs.append(i["img"].unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)

        question = []
        for i in batch:
            question.append(i['question'])
        question = pad_sequence(question, batch_first=True, padding_value=self.pad_idx)

        answer = []
        for i in batch:
            answer.append(i['answer'])

        return imgs, question, answer


class Datamake:
    @staticmethod       
    def get_loader(
        image_path="C:\\Users\\tpath\\Desktop\\nlp\\ri" ,
        vqa_path="C:\\Users\\tpath\\Desktop\\nlp\\data_dictionary.pkl", 
        transforms=None, 
        freq_threshold=1,
        batch_size = 32,
        shuffle = True
    ):
        dataset = VQADataset( vqa_path, image_path, train_vocab=None)
        pad_idx = dataset.vocab.stoi["<PAD>"]
        # print(dataset.vocab.itos)
        loader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            collate_fn=MyCollate(pad_idx=pad_idx))
        return loader, pad_idx, [dataset.vocab.stoi, dataset.vocab.itos]


    def get_loader_val(
        vocab,
        image_path="C:\\Users\\tpath\\Desktop\\nlp\\riv" ,
        vqa_path="C:\\Users\\tpath\\Desktop\\nlp\\data_dictionary_val.pkl", 
        transforms=None, 
        freq_threshold=1,
        batch_size = 32,
        shuffle = True
    ):
        dataset = VQADataset( vqa_path, image_path, train_vocab=vocab)
        pad_idx = dataset.vocab.stoi["<PAD>"]
        loader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            collate_fn=MyCollate(pad_idx=pad_idx))
        return loader, pad_idx, [dataset.vocab.stoi, dataset.vocab.itos]


    def make_ans_dict(path):
        with open( config.TRAIN_DATA_DICT_PATH,"rb") as f:
            x = pickle.load(f)
        a=[]
        for i in range(len(x)):
            tmp = list(x[i].keys())[0] 
            a.append(x[i][tmp]['ans'])

        answers_dict={}
        a = list(set(a))

        for i in range(len(a)):
            answers_dict[a[i]] = i
        
        return answers_dict
