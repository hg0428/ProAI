from nn import NeuralNetwork, sigmoid, sigmoid_derivative
from pickle import dump, load as pload
from atexit import register
from sty import fg, rs
from os import listdir
from os.path import isfile, join
from json import load
import dataloader

models = [f for f in listdir("models") if isfile(join("models", f))]

saveFile = "chat1-new"

# DOESN'T WORK RIGHT WHEN MULTILAYERED.
def save(model):
    dump(model, open(join("models", saveFile), "wb"))
    print("Saved!")


layers = 1
size = 6 * 15
out_size = 6 * 15
architecture = []
for i in range(layers - 1):
    architecture.append((size, [i], "sigmoid"))
architecture.append((out_size, [layers - 1], "sigmoid"))

try:
    with open(join("models", saveFile), "rb") as f:
        ai = pload(f)
except:
    print(architecture, "<- arch")
    ai = NeuralNetwork(size, architecture, save_funct=save)
    print(ai.synaptic_weights, "<- weights")
ai.save_funct = save
mil = ai.input_length
# print(ai.synaptic_weights[20].shape)
tokens = [chr(x) for x in range(46, 65)] + [chr(x) for x in range(91, 127)]
bits_per_character = 6
print(len(tokens))


def tokenizer(x):
    return [tokens.index(i) + 1 for i in x]


def decoder(x):
    return [tokens[i + 1] for i in x]


iters = 1_000_000_000_000
batch_size = 0
learning_rate = 0.02
out = ai.think(
    dataloader.fill(
        dataloader.process_value(
            "hello", bits_per_character, lambda x: tokenizer(x)[0]
        ),
        size,
    )
)
# print(out)
print("out", dataloader.decode(out, bits_per_character, lambda x: decoder([x])[0])[0])
ai.train(
    *dataloader.loadJsonData(
        "chat",
        bits_per_character,
        ai.input_length,
        out_size,
        pre_processor=lambda x: x.lower(),
    ),
    iters,
    batch_size,
    learning_rate,
    log_every=512,
    save_every=512,
    test_on_log=lambda self: print(
        dataloader.decode(
            self.think(
                dataloader.fill(
                    dataloader.process_value(
                        "hello", bits_per_character, lambda x: tokenizer(x)[0]
                    ),
                    size,
                )
            ),
            bits_per_character,
            lambda x: decoder([x])[0],
        )[0]
    ),
)
save(ai)
while True:
    inp = input("> ")
    if inp == "exit":
        break
    out = ai.think(
        dataloader.fill(
            dataloader.process_value(inp, bits_per_character, tokenizer), size
        )
    )
    print(dataloader.decode(out, bits_per_character)[0])
# Remember the Lord.
