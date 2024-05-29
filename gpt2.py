import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import transformers

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


# def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
#     # multi-head causal self attention
#     x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

#     # position-wise feed forward network
#     x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

#     return x
class LayerNorm:
    def __init__(self, g, b):
        self.g = g
        self.b = b

    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x = (x - mean) / np.sqrt(variance + 1e-5)
        return x * self.g + self.b

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    ln1 = LayerNorm(ln_1['weight'], ln_1['bias'])
    ln2 = LayerNorm(ln_2['weight'], ln_2['bias'])
    x = x + mha(ln1(x), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]
    x = x + ffn(ln2(x), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    #x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    #inputs = inputs.reshape(-1, wte.shape[1])  # assuming wte.shape[1] is the embedding dimension
    #inputs = np.pad(inputs, (0, 10))  # pad with zeros to a minimum length of 10
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    print(inputs)
    print(blocks)
   
    # forward pass through n_layer transformer blocks
    for index, block in enumerate(blocks):
        mlp = block['transformer.h.{index}.mlp.c_fc.weight'] #transformer.h.11.mlp.c_proj.weight
        attn = block['transformer.h.{index}.attn.c_proj.weight']
        ln_1 = LayerNorm(block['transformer.h.{index}.ln_1.weight'], block['transformer.h.{index}.ln_1.bias'])
        ln_2 = LayerNorm(block['transformer.h.{index}.ln_2.weight'], block['transformer.h.{index}.ln_1.bias'])
        ln_f_instances = [LayerNorm(ln_1, ln_2) for ln in ln_f]
        x = transformer_block(x, mlp, attn, ln_1, ln_2, n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]
    # ...
    for ln_instance in ln_f_instances:
            x = ln_instance(x)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    import torch
    import numpy as np
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # move this line to the top
    gpt2_params = {'n_head': n_head}  # create a dictionary with expected keyword arguments
    print(gpt2_params)
    #print(inputs)
    
    wte = params['transformer.wte.weight']  # embedding weights
    wpe = params['transformer.wpe.weight']  # positional encoding weights

    blocks=[]
    for param, value in params.items():
        if "transformer.h" in param:
            blocks.append({param:value})
    ln_f=[]
    for param, value in params.items():
        if "transformer.ln_f" in param:
            ln_f.append({param:value})



  # Move all tensors to the GPU (assuming CUDA is available)
    wte = wte.to(device)
    wpe = wpe.to(device)

    # Move each tensor in blocks and ln_f to the device
    for param_dict in blocks:
        for param, value in param_dict.items():
            value.to(device)
    if ln_f:
        for param_dict in ln_f:
            for param, value in param_dict.items():
                value.to(device)
    else:
        print("Warning: ln_f is empty!")

    inputs_tensor = torch.tensor(inputs).to(device) 

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  
        logits = gpt2(inputs_tensor, wte, wpe, blocks, ln_f, **gpt2_params)  
        next_id = np.argmax(logits[-1])  
        inputs.append(int(next_id))  
        inputs_tensor = torch.tensor(inputs).to(device)  

    return inputs[len(inputs) - n_tokens_to_generate :]  

def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    model_name_or_path = "chatA"
    config = GPT2Config.from_pretrained(model_name_or_path, max_length=2048)

    encoder = GPT2Tokenizer.from_pretrained(model_name_or_path) 
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config).cuda()
    # Ensure that the model is properly loaded and initialized
    model.eval()  # or model.train() depending on your use case
    print(config)
    #encoder — это BPE tokenizer, используемый GPT-2:
    hparams = config.to_dict()
    params = list(model.named_parameters())
    #print(params)
    # load encoder, hparams, and params from the released open-ai gpt-2 files

    prompt= "Как дела?"
    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)



    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # convert params to a dictionary
    params_dict = {name: param for name, param in params}

    # generate output ids
    output_ids = generate(input_ids, params_dict, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)