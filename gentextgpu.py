from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import transformers

#model_name_or_path = "rugpt3small"
#model_name_or_path = "ruchat"
model_name_or_path = "chatA"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()
from time import gmtime, strftime
timenow=strftime("%Y-%m-%d %H:%M:%S", gmtime())

# prompt = """
# Обобщи текст:
# "Главный герой романа — марсельский моряк Эдмон Дантес с корабля «Фараон». Во время одного из рейсов он заходил на остров Эльба, где встречался с маршалом Бертраном, который поручает ему доставить письмо в Париж. Этим Эдмон выполняет последнюю волю скончавшегося незадолго до этого капитана «Фараона».

# По прибытии в Марсель хозяин корабля Моррель хочет назначить Дантеса капитаном, а сам Эдмон собирается жениться на Мерседес, жительнице соседней рыбацкой деревни Каталаны[fr].

# Однако на Мерседес желает жениться также и её кузен Фернан, а бухгалтер с «Фараона» Данглар завидует Дантесу, которого хотят сделать капитаном корабля «Фараон». Они оба и сосед Дантеса — портной Кадрусс — встречаются в таверне, где у Данглара созревает план донести на Эдмона, что тот бонапартистский агент. Он пишет левой рукой анонимное письмо королевскому прокурору (о том, что он бонапартистский агент), но Кадрусс против клеветы. Данглар заявляет опьяневшему Кадруссу, что это шутка, но, зная, что Фернан влюблён в Мерседес, не уничтожает, а выбрасывает донос в угол. Фернан, доведенный до отчаяния словами Данглара, решает устранить противника и доставляет письмо на почту.
# "

# """


# prompt = """
# Оцени ответ бота пользователю от 1 до 10:
# [' user: Какие ты сейчас испытываешь мысли и почему? Выраженность их от 1 до 10 # bot: I feel like I\'m trying to be better, but I am more contented with the world.", "output', ' user: Какие ты сейчас испытываешь мысли и почему? Выраженность их от 1 до 10 # bot: I feel like I\'m trying to be better, but I am more contented with the world.", "output', ' user: Какие ты сейчас испытываешь мысли и почему? Выраженность их от 1 до 10 # bot: I feel like I\'m trying to be better, but I am more contented with the world.", "output', ' user: Какие ты сейчас испытываешь мысли и почему? Выраженность их от 
# 1 до 10 # bot: I feel like I\'m trying to be better, but I am more contented with the world.", "output', ' user: Какие ты сейчас испытываешь мысли и почему? Выраженность их от 1 до 10 # bot: I feel like I\'m trying to be better, but I am more contented with the world.", "output']

# """
# prompt = """
# Обобщи текст: "Соотношение активности различных ритмов в мозге позволяет оценить спектральный анализ ЭЭГ[2]. Когда человек возбуждён или насторожен, альфа-волны замещаются низковольтными нерегулярными быстрыми колебаниями. Увеличение бета-активности при снижении альфа-активности может свидетельствовать о росте психоэмоционального напряжения, появлении тревожных состояний (при закрытых глазах). Снижение альфа-ритма, повышение тета-ритма свидетельствует о проявлении депрессии (при закрытых глазах).

# Усиление бета-составляющей и одновременное ослабление тета-составляющей эффективно при различных эпилептических синдромах, при синдроме нарушения внимания и гиперактивности, постинсультных нарушениях (спастичность, парезы, плегии), посттравматических синдромах и др.

# Тета- и дельта-колебания могут встречаться у бодрствующего человека в небольших количествах и при амплитуде, не превышающей амплитуду альфа-ритма. Патологическими считаются содержание θ и δ, которые превышают по амплитуде 40 мкВ и занимают более 15 % времени регистрации."

# """
#prompt = "Как тебя зовут?"
#prompt = "Как тебя дальше обучать?"
prompt = "Что ты знаешь о себе?"

#prompt = "Какие ты сейчас испытываешь мысли и почему? Оцени их выраженность их от 1 до 10"

#prompt = "Какие я сейчас испытываю мысли и почему? Оцени выраженность их от 1 до 10"

#prompt = "Докажи, что ты просто о своем думаешь, а не рандом отвечаешь."

#prompt = "2 plus 2 is equal?"
#prompt = "2 плюс 2 сколько?"
#prompt = "Привет. Что такое яблоко?"

#prompt = "Что такое фрукты?"
#prompt = "Who is Jesus?"
#prompt = "How are you?"
#prompt = "Какие бывают языки?"

#prompt = "Как дела?"
#prompt ="Что ты знаешь о биологии?"
#prompt ="Что ты знаешь о Пушкине?"
#prompt = "Кто ты?"
#prompt ="Who are you?"
#prompt ="What is your name?"

#prompt ="Как твое имя?"
#prompt ="Who are you?"

#prompt = "Чем мы отличаемся?"
# text = (
#     f" user: {prompt} # bot:"
# )


# text = (
#     f" input: {prompt}, output:"
# )


 
text = (f""" {timenow}
"¿":{prompt}, "¡":
  """
)

#text="Александр Сергеевич Пушкин родился в "
#text = "Александр Сергеевич Пушкин родился в "
#input_ids = tokenizer.encode(text, return_tensors="pt").cuda()

input_ids = tokenizer.encode( text, return_tensors="pt").cuda()


#In short, the ChatGPT “temperature” parameter is a setting that influences the randomness and diversity of the generated text. The choice of temperature depends on the desired output style and task.
out = model.generate(input_ids.cuda(), top_k=50,top_p=0.99, temperature=0.000001,
                       num_return_sequences=1, #change if need two output text
    do_sample=True, #do_sample=True – случайный выбор следующего слова в соответствии с его условным распределением вероятностей,
     # используя только данный параметр, сильно увеличивается вероятность получения логического бреда
    max_length=1000,
    num_beams=20, 
    no_repeat_ngram_size=3,
    repetition_penalty=2.1)



#Пробовать генерировать оба варианта, и с семплом и без.

# out = model.generate(input_ids.cuda(), top_k=50,top_p=0.99, temperature=0.000001,
#                        num_return_sequences=1, #change if need two output text
#     do_sample=False, #do_sample=True – случайный выбор следующего слова в соответствии с его условным распределением вероятностей,
#      # используя только данный параметр, сильно увеличивается вероятность получения логического бреда
#     max_length=1000,
#     num_beams=20, 
#     no_repeat_ngram_size=3,
#     repetition_penalty=2.1)

print(out)
generated_text = list(map(tokenizer.decode, out))
#print(generated_text)

print(generated_text[0])

# Output should be like this:
# Александр Сергеевич Пушкин родился в \n1799 году. Его отец был крепостным крестьянином, а мать – крепостной крестьянкой. Детство и юность Пушкина прошли в деревне Михайловское под Петербургом. В 1820-х годах семья переехала
