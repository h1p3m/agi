import os
import sys
import random
from collections import Counter
from typing import Tuple
import requests
import PIL
from PIL import Image
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import seaborn as sns

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Args:
    def __init__(self):
        self.l_text_seq_length = 10
        self.r_text_seq_length = 50
        self.image_tokens_per_dim = 8
        self.image_seq_length = self.image_tokens_per_dim ** 2
        self.epochs = 5
        self.save_path = 'checkpoints/'
        self.model_name = 'awesomemodel_'
        self.save_every = 500
        self.bs = 10
        self.clip = 1.0
        self.lr = 2e-5
        self.wandb = False
        self.lt_loss_weight = 0.01
        self.img_loss_weight = 1
        self.rt_loss_weight = 7
        self.image_size = self.image_tokens_per_dim * 8

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

args = Args()
# Загрузка предобученной модели GPT-2 и токенизатора
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


class FoodDataset(Dataset):
    def __init__(self, file_path, csv_path, shuffle=True):
        # Image transformation
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((args.image_size, args.image_size)),  # Resize images to a fixed size
            T.ToTensor()
        ])
        
        # Read CSV file
        df = pd.read_csv(csv_path, header=None, names=['belok', 'fats', 'uglevod', 'kkal', 'title', 'image_path'])
        
        self.samples = []
        max_text_length = args.l_text_seq_length  # Max text sequence length
        
        # Iterate over rows
        for _, row in df.iterrows():
            caption = f"блюдо: {row['title']}; белков: {row['belok']}; жиров: {row['fats']}; углеводов: {row['uglevod']}; ккал: {row['kkal']};"
            if os.path.isfile(os.path.join(file_path, row['image_path'])):
                text_tokens = gpt_tokenizer.encode(caption)
                # Pad or truncate text tokens
                if len(text_tokens) < max_text_length:
                    # Pad with zeros
                    text_tokens += [0] * (max_text_length - len(text_tokens))
                elif len(text_tokens) > max_text_length:
                    # Truncate
                    text_tokens = text_tokens[:max_text_length]
                self.samples.append([file_path, row['image_path'], text_tokens])
        
        if not self.samples:
            raise ValueError("No valid samples found in the dataset")

        if shuffle:
            random.shuffle(self.samples)
    def __len__(self):
        return len(self.samples)  # Return the length of samples
    def load_image(self, file_path, img_name):
        return PIL.Image.open(os.path.join(file_path, img_name))
    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        item = item % len(self.samples)
        file_path, img_name, text_tokens = self.samples[item]

        try:
            image = self.load_image(file_path, img_name)
            image = self.image_transform(image)
        except Exception as err:
            print(err)
            random_item = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_item)
        
        return torch.tensor(text_tokens), image

dataset = FoodDataset(file_path='C:/multim/content/food/', csv_path='C:/multim/content/food/food.csv')
train_dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, drop_last=True)



captions_num = 5

idx = random.randint(0, len(dataset) - 1)
encoded, image = dataset[idx]

print(encoded)

plt.imshow(image.permute(1, 2, 0).cpu().numpy())
plt.show()

idx = random.randint(0, len(dataset) - 1)
encoded, image = dataset[idx]

print(encoded)

plt.imshow(image.permute(1, 2, 0).cpu().numpy())
plt.show()

df = pd.read_csv('C:/multim/content/food/food.csv')
wc, c = WordCloud(), Counter()

for text in df['name']:
    try:
        c.update(wc.process_text(text))
    except:
        continue

wc.fit_words(c)
plt.figure(figsize=(7, 7))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

text_value_counts = pd.DataFrame(df['name'].value_counts())
ax = sns.histplot(data=text_value_counts, x="name")
ax.set_title('Duplicated text count histogram')
ax.set_xlabel('duplicates count')
plt.show()

img_by_url = 'https://eda.ru/img/eda/c620x415/s1.eda.ru/StaticContent/Photos/160525131253/160602184657/p_O.jpg'
img = Image.open(requests.get(img_by_url, stream=True).raw).resize((args.image_size, args.image_size))
img = T.ToTensor()(img) #3 dimensions for make 2 dimensions  img.unsqueeze(0)

# Adjust image encoder to output tensors of shape (3, 3) or (4, 4)
image_encoder = torch.nn.Sequential(
    torch.nn.Linear(args.image_size * args.image_size * 3, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, args.l_text_seq_length * args.image_tokens_per_dim * args.image_tokens_per_dim),  # Adjust output size
    torch.nn.ReLU()
)
# Create multimodal GPT model

def find_valid_shapes(input_size):
    valid_shapes = []
    for i in range(1, input_size + 1):
        if input_size % i == 0:
            for j in range(1, input_size // i + 1):
                if input_size % (i * j) == 0:
                    k = input_size // (i * j)
                    valid_shapes.append((i, j, k))
    return valid_shapes



class MultimodalGPT(torch.nn.Module):
    def __init__(self, gpt_model, image_encoder):
        super(MultimodalGPT, self).__init__()
        self.gpt = gpt_model
        self.image_encoder = image_encoder

    def forward(self, text, image):
        text_emb = self.gpt.transformer.wte(text)
        image_emb = self.image_encoder(image.view(-1, args.image_size * args.image_size * 3))
        #image_emb= 
        text_emb = text_emb.unsqueeze(1).repeat(1, image_emb.size(1), 1)
        print (text_emb.size())
        # Calculate the shape for reshaping
        batch_size = image_emb.size(0)
        total_elements = image_emb.size(1)
        print (total_elements)
        print (batch_size)
       

        # Determine the number of channels in the reshaped tensor
        num_channels = total_elements // (args.image_size * args.image_size)
        print (num_channels)
        # Ensure that num_channels is non-zero
        if num_channels == 0:
            num_channels = 1

        # Calculate the remaining size after accounting for the channels
        remaining_size = total_elements // (args.image_size * args.image_size * num_channels)
        print (remaining_size)
        # Construct the reshaped size tuple
        #reshaped_size = (batch_size, args.image_size, remaining_size, num_channels)
        reshaped_size = find_valid_shapes(total_elements*batch_size)
        print (reshaped_size)
        # Reshape the image embeddings to match the calculated shape
        #image_emb = image_emb.view(reshaped_size[0])
        #RuntimeError: The size of tensor text_emb (1536) must match the size of tensor image_emb (768) at non-singleton dimension 2

        #rearrenge second dimension to 1536
        #image_emb = image_emb.view(1, args.image_size * 2)
        #delete 3rd dimension
        image_emb= image_emb.unsqueeze(1).repeat(1, text_emb.size(1), 1)
        inputs = torch.cat([text_emb, image_emb], dim=2)
        outputs = self.gpt(inputs_embeds=inputs)
        return outputs.logits



multimodal_gpt = MultimodalGPT(gpt_model, image_encoder).to(device)

# Training
# def train(model, dataloader):
#     optimizer = Adam(model.parameters(), lr=args.lr)
#     criterion = torch.nn.CrossEntropyLoss()
#     model.train()
#     for epoch in range(args.epochs):
#         progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
#         for text, image in progress:
#             optimizer.zero_grad()
#             text = text.to(device)
#             image = image.to(device)
#             outputs = model(text, image)
#             loss = criterion(outputs.view(-1, outputs.size(-1)), text.view(-1))
#             loss.backward()
#             optimizer.step()

# train(multimodal_gpt, train_dataloader)

def train(model, dataloader):
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for text, image in progress:
            optimizer.zero_grad()
            
            text = text[0].to(device)  # Accessing the text tensor from the tuple

            # Tokenize text and pad/truncate to a fixed length

            image = image[0].to(device)
            outputs = model(text, image)
            loss = criterion(outputs.view(-1, outputs.size(-1)), text.view(-1))
            loss.backward()
            print("backward loss was")
            optimizer.step()

train(multimodal_gpt, train_dataloader)

# Generating captions
def generate_captions(img, captions_num, model, tokenizer):
    captions = []
    with torch.no_grad():
        text = torch.zeros(1, args.l_text_seq_length, dtype=torch.long, device=device)
        outputs = model(text, img)

        outputs = outputs.logits
        
        logits = outputs[:, -1, :]
     
        top_probs, top_indices = torch.topk(logits, captions_num)
        for prob, idx in zip(top_probs, top_indices):
            caption = tokenizer.decode(idx, skip_special_tokens=True)
            captions.append((prob.item(), caption))
    return captions

texts = generate_captions(img.unsqueeze(0), captions_num, multimodal_gpt, gpt_tokenizer)
for prob, caption in texts:
    print(f"Probability: {prob:.4f}, Caption: {caption}")



