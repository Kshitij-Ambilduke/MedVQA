import torch 
from mmf.datasets.processors import FastTextProcessor
from mmf.common.registry import registry

# from mmf.utils.build import build_dataset
 
# dataset = build_dataset("hateful_memes")
# dataset.visualize(num_samples=8)

@registry.register_processor("fasttext_sentence_vector")
class FastTextSentenceVectorProcessor(FastTextProcessor):
   # Override the call method
   def __call__(self, item):
       # This function is present in FastTextProcessor class and loads
       # fasttext bin
       self._load_fasttext_model(self.model_file)
       if "text" in item:
           text = item["text"]
       elif "tokens" in item:
           text = " ".join(item["tokens"])
 
       # Get a sentence vector for sentence and convert it to torch tensor
       sentence_vector = torch.tensor( self.model.get_sentence_vector(text),dtype=torch.float)

       # Return back a dict
       return {"text": sentence_vector}

   # Make dataset builder happy, return a random number
   def get_vocab_size(self):
       return None

# All model using MMF need to inherit BaseModel
from mmf.models.base_model import BaseModel
# ProjectionEmbedding will act as proxy encoder for FastText Sentence Vector
from mmf.modules.embeddings import ProjectionEmbedding
# Builder methods for image encoder and classifier
from mmf.utils.build import build_classifier_layer, build_image_encoder
 
@registry.register_model("concat_vl")
class LanguageAndVisionConcat(BaseModel):
 
   def __init__(self, config, *args, **kwargs):
       
       super().__init__(config, *args, **kwargs)
  
   @classmethod
   def config_path(cls):   
       return "/home/roboticslab/Documents/MED-VQA/mmf/mmf/configs/models/concat_vl/default.yaml"
 
   def build(self):
      
       self.vision_module = build_image_encoder(self.config.image_encoder)
 
       self.classifier = build_classifier_layer(self.config.classifier)

       self.language_module = ProjectionEmbedding(**self.config.text_encoder.params)

       self.dropout = torch.nn.Dropout(self.config.dropout)
       
       self.fusion = torch.nn.Linear(**self.config.fusion.params)
       self.relu = torch.nn.ReLU()
 

   def forward(self, sample_list):
       
       text = sample_list["text"]
       image = sample_list["image"]
 
       text_features = self.relu(self.language_module(text))
       image_features = self.relu(self.vision_module(image))
 
       combined = torch.cat( [text_features, image_features.squeeze()], dim=1 )

       fused = self.dropout(self.relu(self.fusion(combined)))
 
       logits = self.classifier(fused)

       output = {"scores": logits}
 
       return output

