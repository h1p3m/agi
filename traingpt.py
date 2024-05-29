from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments  
import transformers


import numpy as np
def shape_tree(d):
    if isinstance(d, tuple):
         return list(d)
    elif isinstance(d, list):
         return [shape_tree(v) for v in d]
    elif isinstance(d, dict):
         return {k: shape_tree(v) for k, v in d.items()}
    else:
         raise ValueError("uh oh")

#print(shape_tree([param.shape for _, param in params]))


def fine_tune_gpt2(model, tokenizer, train_file, output_dir, model_size):

    #model.resize_token_embeddings(len(tokenizer))

        # Get the model's embedding size
    

    # Resize the tokenizer to match the model size
    tokenizer.model_max_length = model_size
    
    # Load training dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=256)  #was 128

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,     #optimal 5 epoch
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("train start")

    trainer.train()
    print("train end")
    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments  
import transformers
import torch.nn as nn
import torch

class FineTuneType:
    FREEZE = 1
    FEATURE_EXTRACTION = 2
    NEW_LAYERS = 3
    CLASSIFIER = 4

class MyModel(nn.Module):
    def __init__(self, backbone, load_pretrained, out_features):
        super().__init__()
        assert backbone in ["chatA", "programming", "bigChat"]
        self.backbone = backbone
        self.pretrained_model = None
        self.tokenizer = None
        self.config=[]
        self.classifier_layers = []
        self.new_layers = []
        
        if backbone == "chatA":
            if load_pretrained:
                print("1 weights weights load")
                self.pretrained_model = GPT2LMHeadModel.from_pretrained("chatA")
            else:
                print("1 weights none")
                self.pretrained_model = GPT2LMHeadModel.from_pretrained("chatA")
            

        elif backbone == "programming":
            if load_pretrained:
                print("2 weights weights load")
                self.pretrained_model = GPT2LMHeadModel.from_pretrained("programming")
            else:
                print("continue2")
                self.pretrained_model = GPT2LMHeadModel.from_pretrained("chatA")
            # end if
            
            self.config = GPT2Config.from_pretrained("chatA", max_length=2048)

            self.tokenizer = GPT2Tokenizer.from_pretrained("chatA")
            self.tokenizer.model_max_length=out_features
            self.pretrained_model = GPT2LMHeadModel.from_pretrained("chatA", config=self.config).cuda()
            # Ensure that the model is properly loaded and initialized
            self.pretrained_model.eval()  # or model.train() depending on your use case
            print(self.config)
            #encoder — это BPE tokenizer, используемый GPT-2:
            hparams = self.config.to_dict()
            params = list(self.pretrained_model.named_parameters())

            #print("HParams:", hparams)
            # print("Params:")
            # for name, param in params:
            #     print(f"{name}: {param.shape}")
            #self.classifier_layers = [self.pretrained_model.fc]
            self.classifier_layers = [self.pretrained_model.lm_head]
           
            self.pretrained_model.lm_head= nn.Linear(
                in_features=1024, out_features=out_features, bias=True
            )
            self.new_layers = [self.pretrained_model.lm_head]
        elif backbone == "bigChat":
            if load_pretrained:
                print("3 weights weights load")
                self.pretrained_model = GPT2LMHeadModel.from_pretrained("bigChat")
            else:
                print("3 weights none")
            # end if
            self.pretrained_model = GPT2LMHeadModel.from_pretrained("chatA")
            
            self.classifier_layers = [self.pretrained_model.lm_head]
            # Replace the final layer with a classifier for 102 classes for the Flowers 102 dataset.
            self.pretrained_model.lm_head = nn.Linear(
                in_features=2048, out_features=out_features, bias=True
            )
            self.new_layers = [self.pretrained_model.lm_head]
    async def fine_tune(self, what: FineTuneType): #freeze
        # The requires_grad parameter controls whether this parameter is
        # trainable during model training.
        m = self.pretrained_model
        for p in m.parameters():
            p.requires_grad = False
        if what is FineTuneType.NEW_LAYERS:
            for l in self.new_layers:
                for p in l.parameters():
                    p.requires_grad = True
        elif what is FineTuneType.CLASSIFIER:
            for l in self.classifier_layers:
                for p in l.parameters():
                    p.requires_grad = True
        else:
            for p in m.parameters():
                p.requires_grad = True
        
        print("start training after freeze")

       

    def get_optimizer_params(self):
        """This method is used only during model fine-tuning when we need to
        set a linearly or exponentially decaying learning rate (LR) for the
        layers in the model. We exponentially decay the learning rate as we
        move away from the last output layer.
        """
        options = []
        if self.backbone == "vgg16":
            # For vgg16, we start with a learning rate of 1e-3 for the last layer, and
            # decay it to 1e-7 at the first conv layer. The intermediate rates are
            # decayed linearly.
            lr = 0.0001
            options.append(
                {
                    "params": self.pretrained_model.classifier.parameters(),
                    "lr": lr,
                }
            )
            final_lr = lr / 1000.0
            diff_lr = final_lr - lr
            lr_step = diff_lr / 44.0
            for i in range(43, -1, -1):
                options.append(
                    {
                        "params": self.pretrained_model.features[i].parameters(),
                        "lr": lr + lr_step * (44 - i),
                    }
                )
            # end for
        elif self.backbone in ["resnet50", "resnet152"]:
            # For the resnet class of models, we decay the LR exponentially and reduce
            # it to a third of the previous value at each step.
            layers = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
            lr = 0.0001
            for layer_name in reversed(layers):
                options.append(
                    {
                        "params": getattr(self.pretrained_model, layer_name).parameters(),
                        "lr": lr,
                    }
                )
                lr = lr / 3.0
            # end for
        # end if
        return options



    # Fine-tune the mode
#fine_tune_gpt2("smallgp2", "C:/multim//mental_health_data.txt", "output")
#fine_tune_gpt2("smallgpt2", "C:/multim/wordsweights.txt", "myassociationrules")
#fine_tune_gpt2("myassociationrules", "C:/multim/ruengtiny.json", "myassociationrules")
#fine_tune_gpt2("myassociationrules", "C:/multim/rugpt_small3.json", "myassociationrules")

#fine_tune_gpt2("rugpt3small", "C:/multim/ruengtiny.json", "ruchat")
#fine_tune_gpt2("ruchat", "C:/multim/ruengtiny.json", "ruchat")

#fine_tune_gpt2("ruchat", "C:/multim/trainsets/rugpt_small3.json", "ruchat")


#fine_tune_gpt2("ruchat", "C:\multim\\trainsets\\ru_sharegpt_cleaned.jsonl", "ruchat")

#fine_tune_gpt2("ruchat", "C:\multim\\trainsets\\intents.json", "ruchat")

#fine_tune_gpt2("ruchat", "C:\multim\\result.json", "ruchat")
#fine_tune_gpt2("ruchat", "C:\multim\\ru_instruct_gpt4.jsonl", "ruchat")

#fine_tune_gpt2("ruchat", "C:\multim\\alpaca_balanced.jsonl", "ruchat")

#fine_tune_gpt2("ruchat", "C:\multim\\ru_turbo_saiga.jsonl", "ruchat")

#fine_tune_gpt2("ruchat", "C:\multim\\optimalset.jsonl", "ruchat")

#fine_tune_gpt2("ruchat", "C:\multim\\shuffled_data.jsonl", "ruchat")
#fine_tune_gpt2("ruchat", "C:\multim\\result.json", "ruchat")


#fine_tune_gpt2("ruchat", "C:\multim\\fixed_data2.jsonl", "ruchat")
#fine_tune_gpt2("ruchat", "C:\multim\\sociation.org.tsv(2)", "chatA")




import json
import random

def shuffle_jsonl(input_file, output_file=None):
    # Чтение всех строк из исходного .jsonl файла
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Перетасовка строк
    random.shuffle(lines)

    # Определение выходного файла
    if output_file is None:
        output_file = input_file

    # Запись перетасованных строк в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)






model_name_or_path = "chatA"


my_model = MyModel("programming", False, 768)

#create backbone layers size
my_model.fine_tune(FineTuneType.NEW_LAYERS)

#after freeze train
fine_tune_gpt2(my_model.pretrained_model, my_model.tokenizer, "C:\multim\\player.txt", "chatB", 768)
#fine_tune_gpt2(my_model.pretrained_model, tokenizer, "C:\multim\\fixed_data3.jsonl", "chatB")

# Пример использования
#input_file = 'data.jsonl'
#shuffle_jsonl(r'C:\multim\fixed_data3.jsonl', 'fixed_data3.jsonl')


print()


#fine_tune_gpt2("chatA", "C:\multim\\fixed_data3.jsonl", "chatA")

#fine_tune_gpt2("chatA", "C:\multim\\math.txt", "chatA")


#fine_tune_gpt2("chatA", "C:\multim\\12asp_chem.txt", "chatA")

#fine_tune_gpt2("chatA", "C:\multim\\player.txt", "chatA")

# fine_tune_gpt2("chatA", "C:\multim\\textbot.py", "chatA")

# fine_tune_gpt2("chatA", "C:\multim\\player.txt", "chatA")
#C:\multim\sociation.org.tsv(2)
