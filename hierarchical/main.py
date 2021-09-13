from __future__ import print_function
import torch 
from _config import Config as config
import model
import trainer
import _utils
from pathlib import Path
import sys
import time
torch.backends.cudnn.benchmark = True

path = Path(config.CHECKPOINT_PATH)
path.mkdir(parents=True, exist_ok=True)
stats_file = open(path / 'stats.txt', 'a', buffering=1)
print(' '.join(sys.argv))
print(' '.join(sys.argv), file=stats_file)

answers_dict = _utils.Datamake.make_ans_dict(config.TRAIN_DATA_DICT_PATH)

train_loader, pad_idx, vocab = _utils.Datamake.get_loader(image_path=config.TRAIN_IMG_PATH, vqa_path=config.TRAIN_DATA_DICT_PATH)
valid_loader, val_pad_idx, val_vocab = _utils.Datamake.get_loader_val(vocab=vocab, image_path=config.TEST_IMG_PATH, vqa_path=config.TEST_DATA_DICT_PATH)

num_classes = len(answers_dict)
vocab_len = len(vocab[0])
coattention = model.CoAttention(num_embeddings=vocab_len, num_classes=num_classes, embed_dim=512, k=64).to(config.DEVICE)    
visualfeatures = model.VisualFeatures(device=config.DEVICE).to(config.DEVICE)
stud = model.Identity(coattention, visualfeatures, vocab_len)

optimizer = torch.optim.Adam(stud.parameters(),lr=config.LR)

if (path / 'checkpoint.pth').is_file():
        ckpt = torch.load(config.CHECKPOINT_PATH +"/"+ 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        stud.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
else:
        start_epoch = config.START_EPOCH 
stud = stud.to(config.DEVICE)
config.START_EPOCH = start_epoch
# print("ok")
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3,verbose=True)

trainer.Trainer.train(stud, train_loader, valid_loader, answers_dict, optimizer, criterion, scheduler, config, stats_file)

