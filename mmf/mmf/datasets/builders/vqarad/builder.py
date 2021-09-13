import json
import logging
import math
import os
import zipfile
from collections import Counter

from mmf.common.constants import VQARAD_DOWNLOAD_URL
from mmf.common.registry import registry
from mmf.datasets.base_dataset_builder import BaseDatasetBuilder
from mmf.datasets.builders.vqarad.dataset import VQARADDataset
from mmf.utils.download import download
from mmf.utils.general import get_mmf_root

logger = logging.getLogger(__name__)


@registry.register_builder("vqarad")
class VQARADBuilder(BaseDatasetBuilder):
    def __init__(self):
        super().__init__("vqarad")
        self.dataset_class = VQARADDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/vqarad/defaults.yaml"


    def build(self, config, dataset_type): 

        download_folder = os.path.join(get_mmf_root(), config.data_dir, config.data_folder)
        # print(get_mmf_root())

        file_name = VQARAD_DOWNLOAD_URL.split("/")[-2]
        local_filename = os.path.join(download_folder, file_name)
        # print(file_name, local_filename)

        # extraction_folder = os.path.join(download_folder, ".".join(file_name.split(".")[-1]))
        extraction_folder = download_folder

        # print(extraction_folder)
        # print(file_name)
        self.data_folder = extraction_folder

        if os.path.exists(local_filename):

            logger.info("VQARAD dataset is already present. Skipping download.")
            return

        if (os.path.exists(extraction_folder) and len(os.listdir(extraction_folder)) != 0):

            return

        logger.info("Donwloading the VQARAD dataset now")
        print(config.data_dir)
        download(VQARAD_DOWNLOAD_URL, download_folder, VQARAD_DOWNLOAD_URL.split("/")[-1])

        logger.info("Downloaded. Extractingn now. This can take time.")
        with zipfile.ZipFile(local_filename, "r") as zip_ref: 
            zip_ref.extractall(download_folder)

    def load(self, config, dataset_type, *args, **kwargs): 
        # print(self.data_folder)
        self.dataset = VQARADDataset(config, data_folder=self.data_folder)
        return self.dataset

    def update_registry_for_model(self, config):    
        registry.register(
        
            self.dataset_name +"_text_vocab_size",
            self.dataset.text_processor.get_vocab_size(), 
            )   
        registry.register(
            self.dataset_name + "_num_final_outputs", 
            self.dataset.answer_processor.get_vocab_size(), 
            )