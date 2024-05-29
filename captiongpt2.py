import requests
import torch
from PIL import Image
from transformers import *
from tqdm import tqdm

import numpy as np

import urllib.parse as parse
import os
from torch.utils.data import DataLoader
from IPython.display import display
import datasets
# set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"



# a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False
    
# a function to load an image
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
        

# a function to perform inference
def get_caption(model, image_processor, tokenizer, image_path):
    image = load_image(image_path)
    # preprocess the image
    img = image_processor(image, return_tensors="pt").to(device)
    # generate the caption (using greedy decoding by default)
    output = model.generate(**img)
    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption


encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"
# the decoder model that process the image features and generate the caption text
# decoder_model = "bert-base-uncased"
# decoder_model = "prajjwal1/bert-tiny"
decoder_model = "ruchat"
# load the model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_model, decoder_model
).to(device)


tokenizer = GPT2TokenizerFast.from_pretrained(decoder_model)
# tokenizer = BertTokenizerFast.from_pretrained(decoder_model)
# load the image processor
image_processor = ViTImageProcessor.from_pretrained(encoder_model)


if "ruchat" in decoder_model:
  # gpt2 does not have decoder_start_token_id and pad_token_id
  # but has bos_token_id and eos_token_id
  tokenizer.pad_token = tokenizer.eos_token # pad_token_id as eos_token_id
  model.config.eos_token_id = tokenizer.eos_token_id
  model.config.pad_token_id = tokenizer.pad_token_id
  # set decoder_start_token_id as bos_token_id
  model.config.decoder_start_token_id = tokenizer.bos_token_id
else:
  # set the decoder start token id to the CLS token id of the tokenizer
  model.config.decoder_start_token_id = tokenizer.cls_token_id
  # set the pad token id to the pad token id of the tokenizer
  model.config.pad_token_id = tokenizer.pad_token_id




# max_length = 32 # max length of the captions in tokens
# coco_dataset_ratio = 50 # 50% of the COCO2014 dataset
# train_ds = datasets.load_dataset("HuggingFaceM4/COCO", split=f"train[:{coco_dataset_ratio}%]")
# valid_ds = datasets.load_dataset("HuggingFaceM4/COCO", split=f"validation[:{coco_dataset_ratio}%]")
# test_ds = datasets.load_dataset("HuggingFaceM4/COCO", split="test")
# len(train_ds), len(valid_ds), len(test_ds)


# # remove the images with less than 3 dimensions (possibly grayscale images)
# train_ds = train_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=2)
# valid_ds = valid_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=2)
# test_ds = test_ds.filter(lambda item: np.array(item["image"]).ndim in [3, 4], num_proc=2)


# def preprocess(items):
#   # preprocess the image
#   pixel_values = image_processor(items["image"], return_tensors="pt").pixel_values.to(device)
#   # tokenize the caption with truncation and padding
#   targets = tokenizer([ sentence["raw"] for sentence in items["sentences"] ], 
#                       max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
#   return {'pixel_values': pixel_values, 'labels': targets["input_ids"]}

# # using with_transform to preprocess the dataset during training
# train_dataset = train_ds.with_transform(preprocess)
# valid_dataset = valid_ds.with_transform(preprocess)
# test_dataset  = test_ds.with_transform(preprocess)


# # a function we'll use to collate the batches
# def collate_fn(batch):
#     return {
#         'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
#         'labels': torch.stack([x['labels'] for x in batch])
#     }


# import evaluate

# # load the rouge and bleu metrics
# rouge = evaluate.load("rouge")
# bleu = evaluate.load("bleu")
  
# def compute_metrics(eval_pred):
#   preds = eval_pred.label_ids
#   labels = eval_pred.predictions
#   # decode the predictions and labels
#   pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
#   labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
#   # compute the rouge score
#   rouge_result = rouge.compute(predictions=pred_str, references=labels_str)
#   # multiply by 100 to get the same scale as the rouge score
#   rouge_result = {k: round(v * 100, 4) for k, v in rouge_result.items()}
#   # compute the bleu score
#   bleu_result = bleu.compute(predictions=pred_str, references=labels_str)
#   # get the length of the generated captions
#   generation_length = bleu_result["translation_length"]
#   return {
#         **rouge_result, 
#         "bleu": round(bleu_result["bleu"] * 100, 4), 
#         "gen_len": bleu_result["translation_length"] / len(preds)
#   }

# num_epochs = 2 # number of epochs
# batch_size = 16 # the size of batches


# # for item in train_dataset:
# #   print(item["labels"].shape)
# #   print(item["pixel_values"].shape)
# #   break

# # define the training arguments
# training_args = Seq2SeqTrainingArguments(
#     predict_with_generate=True,             # use generate to calculate the loss
#     num_train_epochs=num_epochs,            # number of epochs
#     evaluation_strategy="steps",            # evaluate after each eval_steps
#     eval_steps=2000,                        # evaluate after each 2000 steps
#     logging_steps=2000,                     # log after each 2000 steps
#     save_steps=2000,                        # save after each 2000 steps
#     per_device_train_batch_size=batch_size, # batch size for training
#     per_device_eval_batch_size=batch_size,  # batch size for evaluation
#     output_dir="captioning", # output directory
#     # push_to_hub=True # whether you want to push the model to the hub,
#     # check this guide for more details: https://huggingface.co/transformers/model_sharing.html
# )


# # instantiate trainer
# trainer = Seq2SeqTrainer(
#     model=model,                     # the instantiated 🤗 Transformers model to be trained
#     tokenizer=image_processor,       # we use the image processor as the tokenizer
#     args=training_args,              # pass the training arguments
#     compute_metrics=compute_metrics, 
#     train_dataset=train_dataset,     
#     eval_dataset=valid_dataset,      
#     data_collator=collate_fn,        
# )

# from torch.utils.data import DataLoader

# def get_eval_loader(eval_dataset=None):
#   return DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=batch_size)

# def get_test_loader(eval_dataset=None):
#   return DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size)



# Документация рекомендует нам создать подкласс Trainerдля определения нашего собственного тренера, 
# чтобы мы могли выполнять собственное поведение с определенными классами. Поскольку мне для этого слишком лень, 
# я просто переопределяю функции get_training_dataloder(), get_eval_dataloader()и get_test_dataloader()для возврата 
# обычного PyTorch DataLoader:




# def get_eval_loader(eval_dataset=None):
#   return DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=batch_size)

# def get_test_loader(eval_dataset=None):
#   return DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size)

# # override the get_train_dataloader, get_eval_dataloader and
# # get_test_dataloader methods of the trainer
# # so that we can properly load the data
# trainer.get_train_dataloader = lambda: DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size)
# trainer.get_eval_dataloader = get_eval_loader
# trainer.get_test_dataloader = get_test_loader


#print("train start")

# train the model
#trainer.train()


# evaluate on the test_dataset
#trainer.evaluate(test_dataset)
# if __name__ == '__main__':
#  freeze_support()

# using the pipeline API
#image_captioner = pipeline("image-to-text", model="captioning")

image_captioner = pipeline("image-to-text", model="Abdou/vit-swin-base-224-gpt2-image-captioning")
image_captioner.model = image_captioner.model.to(device)

def show_image_and_captions(url):
  # get the image and display it
  display(load_image(url))
  # get the caption
  pipeline_caption = get_caption(image_captioner.model, image_processor, tokenizer, url)
  # print the caption
  print(f"Caption: {pipeline_caption}")


show_image_and_captions("http://images.cocodataset.org/test-stuff2017/000000000001.jpg")