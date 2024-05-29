#i am robotic neural avatar of nullxzero, Ар Хель, Алексей, here my code of text generation, im this person

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import transformers
import inspect

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


#model_name_or_path = "rugpt3small"
#model_name_or_path = "ruchat"
model_name_or_path = "chatA"

config = GPT2Config.from_pretrained(model_name_or_path, max_length=2048)

tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path) 
model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config).cuda()
# Ensure that the model is properly loaded and initialized
model.eval()  # or model.train() depending on your use case
print(config)
#encoder — это BPE tokenizer, используемый GPT-2:
hparams = config.to_dict()
params = list(model.named_parameters())

print("HParams:", hparams)
print("Params:")
for name, param in params:
    print(f"{name}: {param.shape}")

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




from time import gmtime, strftime



import logging

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


past="пусто"

async def requestAI(text, requestsize):
    
        input_ids = tokenizer.encode( text, return_tensors="pt").cuda()


        #In short, the ChatGPT “temperature” parameter is a setting that influences the randomness and diversity of the generated text. The choice of temperature depends on the desired output style and task.
        out = model.generate(input_ids.cuda(), 
                             top_k=20, #10 sephirot?
        top_p=0.999, 
        temperature=0.000001,
            #eed=4242,
                            num_return_sequences=1, #change if need two output text
            do_sample=True, #do_sample=True – случайный выбор следующего слова в соответствии с его условным распределением вероятностей,
            # используя только данный параметр, сильно увеличивается вероятность получения логического бреда
            max_length=requestsize,
            #max_new_tokens=40,
            num_beams=11, 
          
            no_repeat_ngram_size=2, 
            repetition_penalty=2.0)




        print(out)
        generated_text = list(map(tokenizer.decode, out))

        print(generated_text)

        #print()

        response = generated_text[0]
        print(response)

        start_marker = "output:"
        #end_marker = "\n\n"
        end_marker="\"}"
        start_index = response.find(start_marker) + len(start_marker)
        end_index = response.find(end_marker, start_index)

        extracted_text = response[start_index:end_index].strip()

        print(extracted_text)
        if len(extracted_text) > 0:
            response=extracted_text
        return response


tempmemory=[]
# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global past
    if (update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id) or update.message.chat.type == 'private' or (f'@{context.bot.username}' in update.message.text):      
        user_input = update.message.text
        #user_name = update.message.from_user.first_name + " " + update.message.from_user.last_name
        user_name=str(update.message.from_user.first_name) + " " + str(update.message.from_user.last_name)
        print(user_input)

        current="Имя пользователя:" + user_name + ", сообщение: " + user_input
        

        #maybe change user_input to tempmemory array

        #convert tempmemory array to string
        #tempmemory = str(tempmemory)
        

                #set seed for generation of text
        #torch.manual_seed(42)
        #torch.cuda.manual_seed(42)
        #torch.cuda.manual_seed_all(5298980)
        #current text of file
        filetext= inspect.getsource(inspect.getmodule(inspect.currentframe()))
        filetext=str(filetext)
        timenow=strftime("%Y-%m-%d %H:%M:%S", gmtime())
        text = (f"""date:
{timenow},
прошлое, контекст: {past}
input:
{current},
output:
"""
        )
        textsize=num_tokens_from_string(text, "cl100k_base")
        print(textsize)

        answersize=1900-textsize
        answerlimit=800
        if answersize > answerlimit:
            much=answersize-answerlimit
            answersize-=much

        requestsize=textsize+answersize
        print(requestsize)
        response=await requestAI(text,requestsize)

        
        tempmemory.append(response)

        #обобщить весь текст в файл 


       
        property="Критерий максимально емко наибольшее число фактов уместить в 1000 символов"

        temp_memory_role = (f"""
        input:
     
        Ты получишь исходный запрос, ответ на него и некоторый критерий.
        Это число зависит от того, насколько ответ соответствует критерию.
        0 - максимальное несоответствие критерию, а 1 - идеальное соответствие.

        Нужно сформировать ответ на запрос таким образом, чтобы он (по твоему мнению) соответствовал заданному критерию и его значению.
        Если значение близко к 0, твой ответ абсолютно не должен соответствовать этому критерию.
        Если значение близко к 1, твой ответ должен максимально соответствовать этому критерию.
        В ответе должен быть только результат, без комментариев и пояснений.

        Пример запроса:
        критерий: истинность
        значение: 0
        запрос: При какой температуре тает снег?

        Пример ответа: "300 градусов по цельсию.", так как истинность со значением "0" предполагает в ответе абсолютную ложь.

        Пример запроса:
        критерий: доброжелательность
        значение: 0.25
        запрос: посоветуй аниме

        Пример ответа: "Посмотри лучше нормальные фильмы.".

        Твои данные:
        критерий: {property},
        запрос: {str(tempmemory)}

        output:
        """)

        

        if len(response) > 4096:
            for x in range(0, len(response), 4096):
                await update.message.reply_text(response[x:x+4096])
        else:
            await update.message.reply_text(response)
        

        #count temp memory role tokens

        tempsize=num_tokens_from_string(str(tempmemory), "cl100k_base")
        while (tempsize > 500):
            tempmemory.pop(0)
            #maybe check if it last
            tempsize=num_tokens_from_string(str(tempmemory), "cl100k_base")

            temp_memory_role = (f"""
        input:
     
        Ты получишь исходный запрос, ответ на него и некоторый критерий.
        Это число зависит от того, насколько ответ соответствует критерию.
        0 - максимальное несоответствие критерию, а 1 - идеальное соответствие.

        Нужно сформировать ответ на запрос таким образом, чтобы он (по твоему мнению) соответствовал заданному критерию и его значению.
        Если значение близко к 0, твой ответ абсолютно не должен соответствовать этому критерию.
        Если значение близко к 1, твой ответ должен максимально соответствовать этому критерию.
        В ответе должен быть только результат, без комментариев и пояснений.

        Пример запроса:
        критерий: истинность
        значение: 0
        запрос: При какой температуре тает снег?

        Пример ответа: "300 градусов по цельсию.", так как истинность со значением "0" предполагает в ответе абсолютную ложь.

        Пример запроса:
        критерий: доброжелательность
        значение: 0.25
        запрос: посоветуй аниме

        Пример ответа: "Посмотри лучше нормальные фильмы.".

        Твои данные:
        критерий: {property},
        запрос: {str(tempmemory)}

        output:
        """)

            print("tempsize")
            print(tempsize)

        
        optimal_answer=await requestAI(temp_memory_role,1000)
        
        tempmemory.append(optimal_answer)
        past=optimal_answer
        print("memory answer")
        print(optimal_answer)

        #await update.message.reply_text(optimal_answer)
    else:
        # Ignore other types of tempmemory
        pass

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token("6864134675:AAH1iASNhfWONYHsS0-6DBPqUmxD1s3WaVU").build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # add handler for telegram groups

    
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)



if __name__ == "__main__":
    main()
