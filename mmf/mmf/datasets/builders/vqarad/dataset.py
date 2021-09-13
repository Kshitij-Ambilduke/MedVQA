import json
import os

import numpy as np
import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.distributed import is_master, synchronize
from mmf.utils.general import get_mmf_root
from mmf.utils.text import VocabFromText, tokenize
from PIL import Image


_CONSTANTS = {
    "questions_folder": "trainset",
    "dataset_key": "vqarad",
    "empty_folder_error": "VQARAD dataset folder is empty.",
    "questions_key": "question",
    # "question_key": "question",
    "answer_key": "answer",
    "image_key": "image_name",
    # "train_dataset_key": "train",
    "images_folder": "images",  # name of the image_folder is "images" for VQARAD
    "vocabs_folder": "vocabs",
}

_TEMPLATES = {
    "data_folder_missing_error": "Data folder {} for VQARAD is not present.",
    "question_json_file": "trainset.json",
    "vocab_file_template": "{}_{}_vocab.txt",
}

class VQARADDataset(BaseDataset): 

    #data_folder is the json file trainset.json 
    #data_dir is the dir containg the images and the trainset.json 
    #configure the config file of the dataset accordingly 

    def __init__(self, config, data_folder=None, *args, **kwargs): 
        super().__init__(_CONSTANTS["dataset_key"], config)


        self._data_folder = data_folder
        self._data_dir = os.path.join(get_mmf_root(), config.data_dir)   # 

        if not self._data_folder: 
            self._data_folder = os.path.join(self._data_dir, config.data_folder)

        #check if the folder exists 
        if not os.path.exists(self._data_folder): 
            raise RuntimeError(
                _TEMPLATES['data_folder_missing_error'].format(self._data_folder)
                )

        #check if the folder was actually extracted in the subfolder 
        if config.data_folder in os.listdir(self._data_folder):
            self._data_folder = os.path.join(self._data_folder, config.data_folder)

        if len(os.listdir(self._data_folder)) == 0:
            raise FileNotFoundError(_CONSTANTS["empty_folder_error"])

        self.load()

    def load(self):

        self.image_path = os.path.join(
            self._data_folder, _CONSTANTS["images_folder"]) #since only trainset is available , I haven't included the _dataset_type 

        with open(
            os.path.join(
                self._data_folder, 
                _TEMPLATES["question_json_file"]
                )) as f : 
            self.questions = json.load(f)[:]


            if is_master(): 
                self._build_vocab(self.questions, _CONSTANTS["questions_key"])
                self._build_vocab(self.questions, _CONSTANTS["answer_key"])
            synchronize()


    def __len__(self): 
        return len(self.questions)


    def _get_vocab_path(self, attribute): 

        return os.path.join(
            self._data_dir, 
            _CONSTANTS["vocabs_folder"], 
            _TEMPLATES["vocab_file_template"].format(self.dataset_name, attribute), 
        )

    def _build_vocab(self, questions, attribute): 

        # since this contains only training data, vocab file will always be created
        vocab_file = self._get_vocab_path(attribute)

        if os.path.exists(vocab_file): 
            return

        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)

        sentences = [question[attribute] for question in questions]
        
        build_attributes = self.config.build_attributes

        kwargs = {
            "min_count": build_attributes.get("min_count", 1), 
            "keep": build_attributes.get("keep", [";", ","]),
             "remove": build_attributes.get("remove", ["?", "."]),
        }

        if attribute == _CONSTANTS["answer_key"]: 
            kwargs["only_unk_extra"] = False

        # print(sentences)
        vocab = VocabFromText(sentences, **kwargs)

        with open(vocab_file, "w") as f: 
            f.write("\n".join(vocab.word_list))



    def __getitem__(self, idx): 

        data = self.questions[idx]

        current_sample = Sample()
       
        question = data["question"]
        tokens =  tokenize(question, keep=[";", ","], remove=["?", "."])
        processed =  self.text_processor({"tokens": tokens})
        current_sample.text = processed["text"]
        
        processed = self.answer_processor({'answers': [data["answer"]]})
        current_sample.answers = processed["answers"]
        current_sample.targets = processed["answers_scores"]
        #print(processed["answers_scores"])

        #print(type(current_sample.answers))

        image_path = os.path.join(self.image_path, data["image_name"])
        image = np.true_divide(Image.open(image_path).convert("RGB"), 255)
        image = image.astype(np.float32)
        current_sample.image = torch.from_numpy(image.transpose(2, 0, 1))

        return current_sample




        
