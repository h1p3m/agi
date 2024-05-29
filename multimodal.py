import pandas as pd
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer, FlavaModel, PreTrainedTokenizerFast
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments  



class CausalLMOutputWithCrossAttentions:
    def __init__(self, logits, past_key_values, hidden_states, attentions, cross_attentions, multimodal_embeddings):
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.cross_attentions = cross_attentions

        self.multimodal_embeddings = multimodal_embeddings

    def to_tuple(self):
        return self.logits, self.past_key_values, self.hidden_states, self.attentions, self.cross_attentions, self.multimodal_embeddings 
    
    @classmethod
    def from_tuple(cls, tuple):
        return cls(*tuple)
    
    def __getitem__(self, item):
        return self.to_tuple()[item]
    
    def __len__(self):
        return len(self.to_tuple())
    
    def __iter__(self):
        return iter(self.to_tuple())
    
    def __repr__(self):
        return str(self.to_tuple())
    
    def __str__(self):
        return str(self.to_tuple())
    
    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()
    
    def __ne__(self, other):
        return self.to_tuple() != other.to_tuple()
    
    def __hash__(self):
        return hash(self.to_tuple())
    
    def __getstate__(self):
        return self.to_tuple()
    
    def __setstate__(self, state):
        self.logits, self.past_key_values, self.hidden_states, self.attentions, self.cross_attentions, self.multimodal_embeddings = state
    
    def __reduce__(self):
        return (self.__class__, (self.to_tuple(),))
    
    def __reduce_ex__(self, protocol):
        return (self.__class__, (self.to_tuple(),), None, None, None)






dataset = pd.read_csv("C:/multim/content/food/food.csv")
dataset.head()

print(dataset.columns)

image_folder_path = "C:/multim/content/food/"

#add to image name of folder
#dataset['path'] = dataset['name'].apply(lambda x: os.path.join(image_folder_path, x))

#must check missed values
#after test implement 
##dataset = dataset[dataset['text_corrected'].notna() & (dataset['text_corrected'] != '')]

#get in this order

#dataset = dataset.filter(["text_corrected", "image_path", "overall_sentiment"])

#only ,"Белки, г:","Жиры, г:","Углеводы, г:","Калории, ккал:",name,path

#clear all empty values in dataset
dataset = dataset.dropna()

print("==============================================")
print(f'The shape of the dataset is: {dataset.shape}')
print("==============================================")
print(f'The number of sentiments is :\n{dataset.value_counts()}')
print("==============================================")

dataset.head(10)

index = 20

image_path = dataset["path"].iloc[index]
sample_image = Image.open(("content/food/"+image_path))

proteints = dataset["Белки, г:"].iloc[index]
fats = dataset["Жиры, г:"].iloc[index]
carbohydrates = dataset["Углеводы, г:"].iloc[index]

print(f'Proteins: {proteints} g')
print(f'Fats: {fats} g')
print(f'Carbohydrates: {carbohydrates} g')
sample_image



# print(f'The number of sentiments in each category is:\n{dataset.overall_sentiment.value_counts()}')
# print("==============================================")
dataset.head()


dataset = dataset.sample(frac=1, random_state=42)

#split data on parts

train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)

test_data, valid_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print("Train set shape:", train_data.shape)
print("Test set shape:", test_data.shape)
print("Validation set shape:", valid_data.shape)

my_model = GPT2LMHeadModel.from_pretrained("smallgpt2")
#processor = PreTrainedTokenizerFast.from_pretrained("smallgpt2") # it tokenizer too? not fast?
processor = AutoTokenizer.from_pretrained("smallgpt2", add_prefix_space=True, use_fast=False)
# add to tokenizer keyword argument "images"
# processor.encoder = PreTrainedTokenizerFast.from_pretrained("smallgpt2")
# processor.encoder.add_special_tokens({'additional_special_tokens': ['<IMG>']})
#rewrite processor batch_encode_plus method
#need for add images argument
# class PreTrainedTokenizerFast(PreTrainedTokenizerFast):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.encoder = self.get_encoder()
#         self.decoder = self.get_decoder()

#     def get_encoder(self):
#         return self.tokenizer.encoder

#     def get_decoder(self):
#         return self.tokenizer.decoder

#     def __call__(self, text, images, return_tensors, padding):
#         text_encodings = self.encoder(text, return_tensors=return_tensors, padding=padding)
#         image_encodings = self.encoder(images, return_tensors=return_tensors, padding=padding)
#         return text_encodings, image_encodings

#     def __batch_encode_plus(self, text, images, return_tensors, padding):
#         text_encodings = self.encoder.batch_encode_plus(text, return_tensors=return_tensors, padding=padding)
#         image_encodings = self.encoder.batch_encode_plus(images, return_tensors=return_tensors, padding=padding)
#         return text_encodings, image_encodings
#processor.encoder.save_pretrained("smallgpt2")






device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def findTupleShape(tuple):
    return tuple[0].shape
class MultimodalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.max_length = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        #'Белки, г:', 'Жиры, г:', 'Углеводы, г:', 'Калории, ккал:',
        text = self.data.iloc[index][ 'Белки, г:']
        #convert text to string
        text = str(text)
        #show here shape

        #print (text.shape)
        #+ " " + self.data.iloc[index]['Жиры, г:'] + " " + self.data.iloc[index]['Углеводы, г:'] + " " + self.data.iloc[index]['Калории, ккал:']
        image_path = self.data.iloc[index]["path"]
        image = Image.open(("content/food/"+image_path)).convert("RGB")

        labels = self.data.iloc[index][['Unnamed: 0']].values.astype(int)

        inputs = processor(text = text,
                   images=torch.tensor(np.array(image)),
                   return_tensors="pt",
                   padding=True
                   )


        print("inputs:")
        print
        print(inputs)
        #print(inputs.images)
        #{'input_ids': tensor([[657,  13,  23]]), 'attention_mask': tensor([[1, 1, 1]])}
        input_ids = inputs['input_ids'][0]
        #token_type_ids =  inputs['token_type_ids'][0]
        #token type ids - its type of token
        #token_type_ids =  3
       


        attention_mask = inputs['attention_mask'][0]
        #get pixel_values from image
        #convert image to tensor
        #inputs get keyword argument images

        #images=inputs.images
        #print(images)
        pixel_values =torch.tensor(np.array(image))[0].shape
        #convert tuple to shape
        token_type_ids=torch.tensor(findTupleShape(inputs['input_ids']))
        #print(token_type_ids)
        #token_type_ids=pixel_values[0].shape
        #pixel_values =  inputs['pixel_values'][0]
        #inputs['pixel_values'][0] = pixel_values
        print("inputs:")
        print
        print(inputs)
        input_ids = nn.functional.pad(input_ids, (0, self.max_length - input_ids.shape[0]), value=0)
        token_type_ids = nn.functional.pad(token_type_ids, (0, self.max_length - len(token_type_ids)), value=0)
        attention_mask = nn.functional.pad(attention_mask, (0, self.max_length - attention_mask.shape[0]), value=0)

        #return input_ids, token_type_ids, attention_mask, pixel_values , torch.tensor(labels)
        return input_ids,  token_type_ids, attention_mask, pixel_values , torch.tensor(labels)

train_dataset = MultimodalDataset(train_data)
test_dataset = MultimodalDataset(test_data)
val_dataset = MultimodalDataset(valid_data)

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class MultimodalClassifier(nn.Module,):
    def __init__(self, num_labels, my_model):
        super(MultimodalClassifier, self).__init__()
        self.model = my_model
        self.classifier = nn.Sequential(

            nn.Linear(self.model.config.hidden_size, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_labels)

        )

    def forward(self, input_ids, token_type_ids, attention_mask): #, pixel_values
        outputs = self.model(input_ids = input_ids,
                        token_type_ids = token_type_ids,
                        attention_mask = attention_mask
                       # pixel_values = pixel_values
                        )
        
    #get time of model output

        #print(outputs)
        #print(outputs.last_hidden_state.shape)
        #add attribute multimodal_embeddings to model output



        multimodal_embeddings = outputs.multimodal_embeddings
        x = multimodal_embeddings[:, -1, :]
        x = self.classifier(x)
        return x
    
# num_labels = train_data['overall_sentiment'].nunique()
num_labels = train_data["Углеводы, г:"].nunique()


model = MultimodalClassifier(num_labels, my_model).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4)

num_epochs = 10
n_total_steps = len(train_loader)

print (train_loader)

#print type of train_loader
print(type(train_loader))
for epoch in range(num_epochs):

  for i, batch in enumerate (train_loader):

    input_ids, token_type_ids , attention_mask, pixel_values, labels = batch
    input_ids = input_ids.to(device)
    token_type_ids  = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)

    #convert list pixel_values to tensor
    #pixel_values = torch.tensor(pixel_values)
    #TypeError: only integer tensors of a single element can be converted to an index
    pixel_values = torch.tensor(np.array (pixel_values))
    pixel_values = pixel_values.to(device)

    labels = labels.view(-1)
    labels = labels.to(device)

    optimizer.zero_grad()

    logits = model(input_ids = input_ids,
                   token_type_ids = token_type_ids,
                   attention_mask = attention_mask
            #       pixel_values = pixel_values   maybe change model in logits
    )

    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()


    if (i+1) % 16 == 0:
        print(f'epoch {epoch + 1}/ {num_epochs}, batch {i+1}/{n_total_steps}, loss = {loss.item():.4f}')



    


all_labels = []
all_preds = []

with torch.no_grad():
  n_correct = 0
  n_samples = 0
  for i, batch in enumerate (test_loader):

    input_ids, token_type_ids , attention_mask, pixel_values, labels = batch
    input_ids = input_ids.to(device)
    token_type_ids  = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    pixel_values = pixel_values.to(device)

    labels = labels.view(-1)
    labels = labels.to(device)

    optimizer.zero_grad()

    outputs = model(input_ids = input_ids,
                   token_type_ids = token_type_ids,
                   attention_mask = attention_mask,
                   pixel_values = pixel_values
    )

    _, predictions = torch.max(outputs, 1)

    all_labels.append(labels.cpu().numpy())
    all_preds.append(predictions.cpu().numpy())

all_labels = np.concatenate(all_labels, axis=0)
all_preds = np.concatenate(all_preds, axis=0)

print(classification_report(all_labels, all_preds))
print(accuracy_score(all_labels, all_preds))