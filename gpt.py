from nn import NeuralNetwork, to_categorical, AdamW
import numpy as np
import dataloader
from pickle import dump, load as pload
from os.path import isfile, join, exists
from os import makedirs
from unidecode import unidecode
from proai.tokenizer import Tokenizer
import atexit
saveFile = "custom_arch-56l-5in"
input_length = 5

def save(model=None):
    if model == None:
        model = nn
    dump(model, open(join("models", saveFile, 'model.pkl'), "wb"))
    print("Saved!")


def generate_architecture(num_layers): #For my custom arch
    arch = []
    layer_id = 0
    layer_num = 0
    for i in reversed(range(1, num_layers + 1)):
        for x in range(i):
            inps = [0] if layer_num == 0 else list(range(layer_id+layer_num, layer_id+layer_num+i+1))
            arch.append((input_length, inps, "gelu"))
        if layer_num != 0:
            layer_id += i
        layer_num += 1 
    return arch

try:
    tokenizer = Tokenizer.from_save_file(join("models", saveFile, 'tokenizer.json'))
    with open(join("models", saveFile, 'model.pkl'), "rb") as f:
        nn = pload(f)
    print('Loaded.')
except:
    print('Creating')
    newpath = join("models", saveFile)
    if not exists(newpath):
        makedirs(newpath)
    tokens = (
        [""]
        + [chr(x) for x in range(32, 65)]
        + [chr(x) for x in range(91, 127)]
    )
    [tokens.remove(x) for x in "@\\{}|^+#=[]_~`%"]
    architecture = generate_architecture(10)
    architecture.append((len(tokens), [len(architecture)], "softmax"))
    print(architecture, len(architecture))
    print(tokens, len(tokens))
    tokenizer = Tokenizer(tokens)
    tokenizer.save_to_file(join("models", saveFile, 'tokenizer.json'))
    nn = NeuralNetwork(input_length, architecture=architecture, save_funct=save)
    save(nn)


def predict(context):
    context = dataloader.fill(
        tokenizer.tokenize(context), input_length, 0, reverse=True
    )  # input context as a numpy array
    prediction = nn.think(context)  # generate prediction as a numpy array

    predicted_word_index = np.argmax(prediction)
    predicted_word = dataloader.fill(tokenizer.decode([predicted_word_index]), input_length, 0, True)
    prop_dict = dict(zip(tokenizer.vocabulary, prediction))
    # np.random.choice(tokenizer.vocabulary, 1, p=prediction)
    prop_dict = dict(sorted(prop_dict.items(), key=lambda x: x[1], reverse=True))
    print(prop_dict)
    print(list(prop_dict.keys()).index('b'))
    return prop_dict

predict('the quick ')
contexts, next_words = [], []#dataloader.load_data("new_processed_data", 10, 1)
fix = (
    lambda x: unidecode(x.replace("”", '"')
    .replace("“", '"')
    .replace("‘", "'")
    .replace("’", "'")
    .replace("–", "-")
    .replace("—", "-")
    .replace("…", ".")
    .replace("=", " equals ")
    .replace("{", "(")
    .replace("}", ")")
    .replace("[", "(")
    .replace("]", ")")
    .replace("\xa0", " ")
    .replace("›", ">")
    .replace("‹", "<")
    .replace('_', '-')
    .replace('%', ' percent'))
)

def get_training_permutations(string, input_len):
    perm_dict = {}
    for i in range(len(string)):
        for j in range(i+1, min(i+input_len+1, len(string)+1)):
            window = string[i:j]
            if j == len(string):
                perm_dict[window] = ''
            else:
                perm_dict[window] = string[j]
    return perm_dict

training_data = {}
training_data = get_training_permutations('the quick brown', 5)
contexts = [
    dataloader.fill(tokenizer.tokenize(context), input_length, 0, reverse=True)
    for context in contexts + list(training_data.keys())]
print(f"Contexts loaded. Training set size: {len(contexts)}")
next_words = to_categorical(
    [dataloader.fill(tokenizer(next_word), 1)[0] for next_word in next_words + list(training_data.values())],
    len(tokenizer.vocabulary),
)
print('Ready to begin training. Constructing Optimizers...')
start_lr = 0.01
def learning_rate_scheduler(lr, epoch):
    if epoch < 10:
        return start_lr
    elif epoch < 20:
        return start_lr * 0.1
    elif epoch < 30:
        return start_lr * 0.01
    else:
        return start_lr * 0.001

atexit.register(save)
optimizers = []
for i in range(len(nn.architecture)):
    optimizers.append(AdamW(beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0))
print("Optimizers constructed. Converting data to numpy arrays...")    
nn.train( # Takes 24 (probably a lot more now) min per iter for GPT-50l-50in, Takes 381.36 hours (probably more now) for custom_arch-56l-50in
    contexts,
    next_words,
    training_epochs=10000,
    batch_size=1,
    learning_rate=start_lr,
    log_every=100,
    save_every=None,
    test_on_log=lambda x: predict('the quick '),
    lambda_val=0,
    max_adjustment_norm=3,
    learning_rate_schedule=learning_rate_scheduler,
    optimizers=None#optimizers
)

