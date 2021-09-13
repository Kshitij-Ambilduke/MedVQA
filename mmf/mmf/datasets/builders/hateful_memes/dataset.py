# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import os

import numpy as np
import omegaconf
import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.general import get_mmf_root
from mmf.utils.visualize import visualize_images
from PIL import Image
from torchvision import transforms
from mmf.utils.distributed import is_master, synchronize
from mmf.utils.text import VocabFromText, tokenize

_CONSTANTS = {
    "vocabs_folder": "vocabs"
}

_TEMPLATES = {
    "vocab_file_template": "{}_{}_vocab.txt"
}

class HatefulMemesFeaturesDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="hateful_memes", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        assert (
            self._use_features
        ), "config's 'use_images' must be true to use image dataset"
        self.is_multilabel = self.config.get("is_multilabel", False)

    def preprocess_sample_info(self, sample_info):
        image_path = sample_info["img"]
        # img/02345.png -> 02345
        feature_path = image_path.split("/")[-1].split(".")[0]
        # Add feature_path key for feature_database access
        sample_info["feature_path"] = f"{feature_path}.npy"
        return sample_info

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info = self.preprocess_sample_info(sample_info)

        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        if "input_ids" in processed_text:
            current_sample.update(processed_text)

        # current_sample.id = torch.tensor(int(sample_info["id"]), dtype=torch.int)
        current_sample.id = torch.tensor(1, dtype=torch.int)

        # Instead of using idx directly here, use sample_info to fetch
        # the features as feature_path has been dynamically added
        features = self.features_db.get(sample_info)
        if hasattr(self, "transformer_bbox_processor"):
            features["image_info_0"] = self.transformer_bbox_processor(
                features["image_info_0"]
            )
        current_sample.update(features)

        fg_dataset_type = self.config.get("fg_dataset_type", None)
        if fg_dataset_type:
            current_sample = self.process_fg_labels(
                fg_dataset_type=fg_dataset_type,
                sample_info=sample_info,
                current_sample=current_sample,
            )
        else:
            if "label" in sample_info:
                current_sample.targets = torch.tensor(
                    sample_info["label"], dtype=torch.long
                )

        return current_sample

    def process_fg_labels(self, fg_dataset_type, sample_info, current_sample):
        """
        If fg_dataset_type is present, it means we are using
        the Hateful Memes Fine Grained datasets. It is the same
        hateful memes datasets but have additional labels for
        protected groups and attack vectors. More details see:
        https://github.com/facebookresearch/fine_grained_hateful_memes
        """
        ds_type_to_label = {
            "attack": sample_info["top_attacks"],
            "pc": sample_info["top_protectedcats"],
            "pc_attack": sample_info["top_protectedcats"] + sample_info["top_attacks"],
            "hateful_pc_attack": sample_info["top_protectedcats"]
            + sample_info["top_attacks"]
            + ["hateful" if int(sample_info["label"]) == 1 else "not_hateful"],
        }
        processed = self.answer_processor(
            {"answers": ds_type_to_label[fg_dataset_type]}
        )
        current_sample.answers = processed["answers"]
        current_sample.targets = processed["answers_scores"]

        return current_sample

    def format_for_prediction(self, report):
        if self.is_multilabel:
            return generate_multilabel_prediction(report)
        else:
            return generate_binary_prediction(report)


class HatefulMemesImageDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="hateful_memes", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        assert (
            self._use_images
        ), "config's 'use_images' must be true to use image dataset"
        self.is_multilabel = self.config.get("is_multilabel", False)
        self.load()

    def init_processors(self):
        super().init_processors()
        # Assign transforms to the image_db
        self.image_db.transform = self.image_processor

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        if "input_ids" in processed_text:
            current_sample.update(processed_text)

        # current_sample.id = torch.tensor(int(sample_info["id"]), dtype=torch.int)
        current_sample.id = torch.tensor(1, dtype=torch.int)

        # Get the first image from the set of images returned from the image_db
        current_sample.image = self.image_db[idx]["images"][0]

        if "label" in sample_info:
            current_sample.targets = torch.tensor(
                sample_info["label"], dtype=torch.long
            )

        return current_sample

    def format_for_prediction(self, report):
        if self.is_multilabel:
            return generate_multilabel_prediction(report)
        else:
            return generate_binary_prediction(report)

    def _get_vocab_path(self, attribute): 

        return os.path.join(
            "/home/roboticslab/Documents/MED-VQA/dataset/med-vqa-data/",      
            _CONSTANTS["vocabs_folder"], 
            _TEMPLATES["vocab_file_template"].format("VQA_MED_2019", 'TRAIN'), 
        )

    def _build_vocab(self, annotation_db, attribute='TRAIN'): 

        # since this contains only training data, vocab file will always be created
        vocab_file = self._get_vocab_path('TRAIN')

        if os.path.exists(vocab_file): 
            return

        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)

        sentences = [question['text'] for question in annotation_db]
        
        build_attributes = self.config.build_attributes

        kwargs = {
            "min_count": build_attributes.get("min_count", 1), 
            "keep": build_attributes.get("keep", [";", ","]),
             "remove": build_attributes.get("remove", ["?", "."]),
        }

        # if attribute == _CONSTANTS["answer_key"]: 
        #     kwargs["only_unk_extra"] = False

        # print(sentences)
        vocab = VocabFromText(sentences, **kwargs)

        with open(vocab_file, "w") as f: 
            f.write("\n".join(vocab.word_list))


    def load(self):

        # self.image_path = os.path.join(
        #     config.data_dir, _CONSTANTS["images_folder"]) #since only trainset is available , I haven't included the _dataset_type 

        # with open(
        #     os.path.join(
        #         self._data_folder, 
        #         _TEMPLATES["question_json_file"]
        #         )) as f : 
        #     self.questions = json.load(f)[:]


        if is_master(): 
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            self._build_vocab(self.annotation_db, 'TRAIN')
        synchronize()

    def visualize(self, num_samples=1, use_transforms=False, *args, **kwargs):
        image_paths = []
        random_samples = np.random.randint(0, len(self), size=num_samples)

        for idx in random_samples:
            image_paths.append(self.annotation_db[idx]["img"])

        images = self.image_db.from_path(image_paths, use_transforms=use_transforms)
        visualize_images(images["images"], *args, **kwargs)




def generate_binary_prediction(report):
    scores = torch.nn.functional.softmax(report.scores, dim=1)
    _, labels = torch.max(scores, 1)
    # Probability that the meme is hateful, (1)
    probabilities = scores[:, 1]

    predictions = []

    for idx, image_id in enumerate(report.id):
        proba = probabilities[idx].item()
        label = labels[idx].item()
        predictions.append({"id": image_id.item(), "proba": proba, "label": label})
    return predictions


def generate_multilabel_prediction(report):
    scores = torch.sigmoid(report.scores)
    return [
        {"id": image_id.item(), "scores": scores[idx].tolist()}
        for idx, image_id in enumerate(report.id)
    ]
