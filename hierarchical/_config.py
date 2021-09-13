class Config:
    LR = 1e-4
    MAX_EPOCHS = 200

    TRAIN_IMG_PATH = "/home/roboticslab/Documents/MED-VQA/dataset/med-vqa-data/med-vqa-data-main-vqa-med-2019/Resize_images"
    TEST_IMG_PATH = "/home/roboticslab/Documents/MED-VQA/dataset/med-vqa-data/med-vqa-data-main-vqa-med-2019/Resize_images_val"
    # TRAIN_IMG_PATH = TEST_IMG_PATH

    TRAIN_DATA_DICT_PATH = "/home/roboticslab/Documents/MED-VQA/dataset/med-vqa-data/med-vqa-data-main-vqa-med-2019/data_dictionary.pkl"
    TEST_DATA_DICT_PATH =  "/home/roboticslab/Documents/MED-VQA/dataset/med-vqa-data/med-vqa-data-main-vqa-med-2019/data_dictionary_val.pkl"
    # TRAIN_DATA_DICT_PATH = TEST_DATA_DICT_PATH
    
    LOSS_PATH = "home/terasquid/Documents/med-VQA/baselines/Heirarchical/loss_of_epoch1.txt"
    MODEL_STORE_PATH = "/home/roboticslab/Documents/MED-VQA/Med-VQA"
    DEVICE = 'cuda'

    CHECKPOINT_PATH = "/home/roboticslab/Documents/MED-VQA/Med-VQA"
    START_EPOCH = 0



