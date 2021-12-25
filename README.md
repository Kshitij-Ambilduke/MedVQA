# Med-VQA

In this repository we have tested 3 VQA models on the ImageCLEF-2019 dataset. Two of these are made on top of Facebook AI Reasearch's [Multi-Modal Framework (MMF)](https://mmf.sh/). 


|Model Name| Accuracy| Number of Epochs|
|-------|------|-------|
|[Hierarchical Question-Image Co-attention](https://arxiv.org/abs/1606.00061) | 48.32% | 42 | 
| MMF Transformer | 51.76% | 30 | 
| [MMBT](https://arxiv.org/abs/1909.02950) | 86.78% | 30 | 

## Test them for yourself!

Download the dataset from [here](https://gitlab.com/aneesh-shetye/med-vqa-data/-/tree/main/vqa-med-2019) and place it in a directory named `/dataset/med-vqa-data/` in the directory where this repository is cloned. 

### MMF Transformer:

```bash 
mmf_run config=projects/hateful_memes/configs/mmf_transformer/defaults.yaml     model=mmf_transformer     dataset=hateful_memes training.checkpoint_interval=100 training.max_updates=3000
```
### MMBT: 

```bash
mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml     model=mmbt     dataset=hateful_memes training.checkpoint_interval=100 training.max_updates=3000
```
### Heirarchical Question-Image Co-attention: 

```bash 
cd hierarchical \ 
python main.py
```

### Dataset details:
Dataset used for training the models was the **VQA-MED** dataset taken from "ImageCLEF 2019: Visual Question Answering in Medical Domain" competition. Following are few plots of some statistics of the dataset.
| ![](https://github.com/Kshitij-Ambilduke/MedVQA/blob/main/dataset_plots/total%20question%20types.PNG) | 
|:--:| 
| *Distribution of the type of questions in the dataset.* |

| ![](https://github.com/Kshitij-Ambilduke/MedVQA/blob/main/dataset_plots/frequencies_better.PNG) | 
|:--:| 
| *Plot of frequency of words in answer.* |
