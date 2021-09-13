import math
import numpy as np
from _config import Config as config
import torch
import sys
import json
import torch
import time

class Trainer:
    @staticmethod
    
    def train(model, train_loader, valid_loader, answers_dict, optimizer, criterion, scheduler, config, stats_file):
        epoch_loss = []
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        # print(f"Total parameters = {pytorch_total_params}")
        print(config.START_EPOCH)
        for n in range(config.START_EPOCH, config.MAX_EPOCHS):
            start_time = time.time()
            model.train()
            l = 0
            for step,i in enumerate(train_loader):
                images = i[0].permute(0, 3, 1, 2).to(config.DEVICE)
                question = i[1].long().to(config.DEVICE)
                answers = i[2]
                target=[]

                for i in answers:
                    target.append(answers_dict[i])

                target=torch.Tensor(target).long().to(config.DEVICE)
                out = model(images, question)
                optimizer.zero_grad()
                loss = criterion(out, target)
                loss.backward()
                # print("hello world")
                l+=loss.item()
                optimizer.step()
        
            l /= len(train_loader)
            # loss_file = open(config.LOSS_PATH,"a")
            # loss_file.write(f"Loss on epoch {n} : {str(l)} \n")
            # loss_file.close()
            scheduler.step(l)
            # torch.save(model.state_dict(), config.MODEL_STORE_PATH+str(n)+".pth")
            
            stats = dict(epoch=n, 
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)
            k = Trainer.accuracy(model, valid_loader, answers_dict, config)
            # loss_file = open(config.LOSS_PTH)
            # loss_file.write(f"Accuracy on epoch {n} : {k} \n")
            # loss_file.close()
            print(f"Accuracy on epoch {n} : {k}")
            state = dict(epoch=n + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, config.CHECKPOINT_PATH+"/"+ 'checkpoint.pth')
            # print(l)
            epoch_loss.append(l)

        return epoch_loss


    @staticmethod
    def accuracy(model, valid_loader, answers_dict, config):
        with torch.no_grad():
            k=0
            for n, i in enumerate(valid_loader):
                images = i[0].permute(0, 3, 1, 2).to(config.DEVICE)
                question = i[1].long().to(config.DEVICE)
                answers = i[2]

                target=[]
                tmp=[]

                for i in answers:
                    tmp.append(answers_dict[i])
                
                target = torch.Tensor(target).long().to(config.DEVICE)
                out = model(images, question)
                out = torch.softmax(out, -1)
                out = torch.argmax(out, dim=1)
                tmp = np.array(tmp)
                out = np.array(out.tolist())
                k += np.sum(tmp==out)

        return k
            
