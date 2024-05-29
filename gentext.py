from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

import chaotictext 
# download & load GPT-2 model
#gpt2_generator = pipeline('text-generation', model='smallgpt2')
gpt2_generator = pipeline('text-generation', model='myassociationrules')
#save the model
#gpt2_generator.save_pretrained('smallgpt2')


#show pipeline layers of model
print(gpt2_generator.model.config)

#change layers(?) of model
#gpt2_generator.model.config.max_length = 512


#do_sample - if true, then model will generate text

#temperature - is probability is high token
#top_k count highest probability tokens
#max_length - maximum length of the generated text
#num_return_sequences - number of sentences to generate


#need to find topology of max train text
def find_valid_shapes(input_size):
    valid_shapes = []
    for i in range(1, input_size + 1):
        if input_size % i == 0:
            for j in range(1, input_size // i + 1):
                if input_size % (i * j) == 0:
                    k = input_size // (i * j)
                    valid_shapes.append((i, j, k))
    return valid_shapes


prompt = "What can you do?"
#prompt = "How are you?"
#prompt = "Как дела?"
formatted_prompt = (
    f"### message: {prompt} ### response:"
)


sentences = gpt2_generator(formatted_prompt, do_sample=True, top_k=50, temperature=0.6, max_length=512, num_return_sequences=3)
for sentence in sentences:
    #print(sentence["generated_text"])
    print(sentence)
    print("="*50)

def harmonic_mean(arr):
        return len(arr) / sum(1 / i for i in arr)

def talk(formatted_prompt):

    question= chaotictext.ngrams(formatted_prompt, 3)

    myselfpuppet = chaotictext.ngrams(__file__, 3)

    sentences = gpt2_generator(formatted_prompt, do_sample=True, top_k=50, temperature=0.6, max_length=512, num_return_sequences=3)
    for sentence in sentences:
    #print(sentence["generated_text"])
        print(sentence)
        print("="*50)

    #sentences all messages?
    print("последнее сообщение:")

    answer=sentence["generated_text"][0]
    print(answer)

    answerface = chaotictext.ngrams(answer, 3)

    corpus = [(formatted_prompt, question), 
        (__file__, myselfpuppet), 
        (answer, answerface)]

    vocab = []

    #append array to array
    vocab. append(chaotictext.ngrams(formatted_prompt, 1))
    vocab. append(chaotictext.ngrams(__file__, 1))
    vocab. append(chaotictext.ngrams(answer, 1))

    #add 
    deal = chaotictext.tfidf(corpus, vocab)

    print(deal)


#harmonic function for each row in array, if 1 (myselfpuppet) bigger then 0, then reroll answer
    if harmonic_mean(deal[1]) > harmonic_mean(deal[0]):
        print("reroll answer")
        talk(formatted_prompt)
    else:
        print("answer is valid")
        #maybe write time in graph modal mind
        #timenow = datetime.datetime.now()
        #modalmind.activity_forecast(timenow, 2)


#print(harmonic_mean(deal))
