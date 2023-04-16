from nn import NeuralNetwork, to_categorical
import numpy as np
import dataloader
from pickle import dump, load as pload
from os.path import isfile, join
print(np.finfo(np.longdouble).max)
saveFile = "gpt-2l-20in"


def save(model):
    dump(model, open(join("models", saveFile), "wb"))
    print("Saved!")


def tokenizer(x):
    return [tokens.index(i) for i in x]


def decoder(x):
    return [tokens[i] for i in x]


tokens = (
    [""]
    + [chr(x) for x in range(32, 65)]
    + [chr(x) for x in range(91, 127)]
    + list("\nœàâé")
)

[tokens.remove(x) for x in "@\\{}|^+#=[]"]
print(tokens, len(tokens))

input_length = 20
architecture = [(20, [0], "bounded_gelu"), (20, [0], "bounded_gelu"), (len(tokens), [1], "softmax")]
try:
    with open(join("models", saveFile), "rb") as f:
        nn = pload(f)
    print('Loaded.')
except:
    print('Creating')
    nn = NeuralNetwork(input_length, architecture=architecture, save_funct=save)

context = dataloader.fill(
    tokenizer("the quick "), input_length, 0, reverse=True
)  # input context as a numpy array
prediction = nn.think(context)  # generate prediction as a numpy array

predicted_word_index = np.argmax(prediction)
predicted_word = dataloader.fill(decoder([predicted_word_index]), input_length, 0, True)
prop_dict = dict(zip(tokens, prediction))
print(prediction, dict(sorted(prop_dict.items(), key=lambda x: x[1], reverse=True)), np.random.choice(tokens, 1, p=prediction))

contexts, next_words = dataloader.load_data("processed_data", 10, 1)
fix = (
    lambda x: x.replace("”", '"')
    .replace("“", '"')
    .replace("‘", "'")
    .replace("’", "'")
    .replace("–", "-")
    .replace("—", "-")
    .replace("…", ".")
    .replace("è", "e")
    .replace("=", " equals ")
    .replace("{", "(")
    .replace("}", ")")
    .replace("[", "(")
    .replace("]", ")")
    .replace("\xa0", " ")
    .replace("›", "<")
    .replace("‹", "<")
)
contexts = [
    dataloader.fill(tokenizer(fix(context)), input_length, 0, reverse=True)
    for context in contexts
]
print("Contexts loaded.")
next_words = to_categorical(
    [dataloader.fill(tokenizer(fix(next_word)), 1)[0] for next_word in next_words],
    len(tokens),
)
nn.train(
    contexts,
    next_words,
    training_iterations=10000,
    batch_size=2**14, #CAN BE INCREASED!
    learning_rate=0.0001,
    log_every=64,
    save_every=64,
    lambda_val=0.1,
    max_adjustment_norm=100
)


num_predictions = 10  # number of predictions to generate
predictions = []

for i in range(num_predictions):
    prediction = np.squeeze(nn.think(context))
    predicted_word_index = np.random.multinomial(1, prediction).argmax()
    predicted_word = decoder([predicted_word_index])
    predictions.append(predicted_word)
