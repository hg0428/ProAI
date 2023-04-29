#WATCH CHURCH SERVICE FIRST

from nn import NeuralNetwork, to_categorical, AdamW
import numpy as np
import dataloader
from pickle import dump, load as pload
from os.path import isfile, join
from unidecode import unidecode
from proai.tokenizer import Tokenizer
import atexit
saveFile = "GPT-50l-50in"
input_length = 50

def save(model=None):
    if model == None:
        model = nn
    dump(model, open(join("models", saveFile, 'model.pkl'), "wb"))
    print("Saved!")


# tokens = (
#     [""]
#     + [chr(x) for x in range(32, 65)]
#     + [chr(x) for x in range(91, 127)]
# )

# [tokens.remove(x) for x in "@\\{}|^+#=[]_~`%"]
# print(tokens, len(tokens))
#try:
tokenizer = Tokenizer.from_save_file(join("models", saveFile, 'tokenizer.json'))
with open(join("models", saveFile, 'model.pkl'), "rb") as f:
    nn = pload(f)
print('Loaded.')
# except:
#     print('Creating')
#     nn = NeuralNetwork(input_length, architecture=architecture, save_funct=save)


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
    return prop_dict

predict('the quick ')
contexts, next_words = dataloader.load_data("new_processed_data", 10, 1)
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
training_data = get_training_permutations('the quick brown fox jumps over a lazy dog', 5)
to_categorical([1], 55)
contexts = [
    dataloader.fill(tokenizer.tokenize(context), input_length, 0, reverse=True)
    for context in contexts + list(training_data.keys())]
print(f"Contexts loaded. Training set size: {len(contexts)}")
next_words = to_categorical(
    [dataloader.fill(tokenizer(next_word), 1)[0] for next_word in next_words + list(training_data.values())],
    len(tokenizer.vocabulary),
)
print('Ready to begin training. Constructing Optimizers...')
def learning_rate_scheduler(lr, epoch):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.1
    elif epoch < 30:
        return lr * 0.01
    else:
        return lr * 0.001

atexit.register(save)
optimizers = []
for i in range(len(nn.architecture)):
    optimizers.append(AdamW(beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01))
print("Optimizers constructed. Converting data to numpy arrays...")    
nn.train( # Takes 24 min per iter
    contexts,
    next_words,
    training_epochs=1000,
    batch_size=16,
    learning_rate=3,
    log_every=100,
    save_every=None,
    test_on_log=None, #lambda x: [predict("hello"), predict('ello ')],
    lambda_val=0.04,
    max_adjustment_norm=10,
    learning_rate_schedule=learning_rate_scheduler,
    optimizers=optimizers
)


num_predictions = 10  # number of predictions to generate
predictions = []

for i in range(num_predictions):
    prediction = np.squeeze(nn.think(context))
    predicted_word_index = np.random.multinomial(1, prediction).argmax()
    predicted_word = tokenizer.decode([predicted_word_index])
    predictions.append(predicted_word)
