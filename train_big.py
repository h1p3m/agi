from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments  


def fine_tune_gpt2(model_name, train_file, output_dir):
    # Load GPT-2 model and tokenizer
    config = GPT2Config.from_pretrained(model_name)


    print(config)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    model = GPT2LMHeadModel.from_pretrained(model_name, config=config).cuda()
    print(config)

    # Load training dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=256)

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

    

    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)







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

# Пример использования
#input_file = 'data.jsonl'
# shuffle_jsonl(r'C:\multim\fixed_data3.jsonl', 'fixed_data3.jsonl')







# fine_tune_gpt2("chatA", "C:\multim\\fixed_data3.jsonl", "chatA")




fine_tune_gpt2("chatA", "C:\multim\\player.txt", "chatA")
shuffle_jsonl(r'C:\multim\fixed_data3.jsonl', 'fixed_data3.jsonl')

fine_tune_gpt2("chatA", "C:\multim\\12asp_chem.txt", "chatA")
fine_tune_gpt2("chatA", "C:\multim\\physics.txt", "chatA")
fine_tune_gpt2("chatA", "C:\multim\\math.txt", "chatA")


fine_tune_gpt2("chatA", "C:\multim\\pythoncoding.txt", "chatA")
fine_tune_gpt2("chatA", "C:\multim\\math.txt", "chatA")


fine_tune_gpt2("chatA", "C:\multim\\poetry.txt", "chatA")
#fine_tune_gpt2("chatA", "C:\multim\\textbot.py", "chatA")



fine_tune_gpt2("chatA", "C:\multim\\textbot.py", "chatA")
fine_tune_gpt2("chatA", "C:\multim\\math.txt", "chatA")
fine_tune_gpt2("chatA", "C:\multim\\fixed_data3.jsonl", "chatA")
#fine_tune_gpt2("chatA", "C:\multim\\result.json", "chatA") #telegram messages




#fine_tune_gpt2("chatA", "C:\multim\\fixed_data3.jsonl", "chatA")

# fine_tune_gpt2("chatA", "C:\multim\\12asp_chem.txt", "chatA")

# fine_tune_gpt2("chatA", "C:\multim\\player.txt", "chatA")

# fine_tune_gpt2("chatA", "C:\multim\\math.txt", "chatA")


# fine_tune_gpt2("chatA", "C:\multim\\player.txt", "chatA")
# fine_tune_gpt2("chatA", "C:\multim\\math.txt", "chatA")
# fine_tune_gpt2("chatA", "C:\multim\\math.txt", "chatA")
# fine_tune_gpt2("chatA", "C:\multim\\physics.txt", "chatA")
# fine_tune_gpt2("chatA", "C:\multim\\physics.txt", "chatA")
# fine_tune_gpt2("chatA", "C:\multim\\math.txt", "chatA")
# fine_tune_gpt2("chatA", "C:\multim\\player.txt", "chatA")

# fine_tune_gpt2("chatA", "C:\multim\\pythoncoding.txt", "chatA")




#C:\multim\sociation.org.tsv(2)
