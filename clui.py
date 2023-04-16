from nn import NeuralNetwork, decode, sigmoid, sigmoid_derivative
from pickle import dump, load as pload
from atexit import register
from sty import fg, rs
from os import listdir
from os.path import isfile, join
from json import load
import dataloader
ai = None
mil = 20  # max input length
mol = 20  # max output length
bpc = 8  # bits per character
layers = 1  

models = [f for f in listdir("models") if isfile(join("models", f))]

saveFile = ""


def save(model):
    print("Saved!")
    dump(model, open(join("models", saveFile), "wb"))


def loadTrainingData(x):
    with open(f"training_data/{x}.json") as f:
        data = load(f)
    return list(data.keys()), list(data.values())


while True:
    print(f'Avialiable models: {", ".join(models)}')
    saveFile = input("Which model should be loaded? ")
    if saveFile in models:
        try:
            ai = pload(open(join("models", saveFile), "rb"))
            if 'activation_func' not in dir(ai):
                ai.activation_func = sigmoid
                ai.activation_func_derivative = sigmoid_derivative
            if 'save_funct' not in dir(ai) or not ai.save_funct:
                ai.save_funct = save
            if 'type' not in dir(ai):
                ai.type = 'NN'
            bpc = ai.bits_per_character
            mil = int(ai.input_length / bpc)
            mol = int(ai.output_length / bpc)
            layers = ai.layers
            t = ai.type
            print(
                f'Loaded {t} "{saveFile}" {layers}l - {mil}:{mol}@{bpc} = {((mol*bpc)*(mil*bpc))*layers:,}'
            )
            break
        except Exception as e:
            print(
                "Error loading save file:",
                dir(e),
                e.__cause__,
                e.args,
                e.__traceback__,
                e,
            )

    else:
        x = input("Would you like to create a new model? ").lower()
        if x.startswith("y"):
            try:
                mil = int(input("Max input length? "))
            except:
                continue
            try:
                mol = int(input("Max output length? "))
            except:
                continue
            try:
                bpc = int(input("Bits per character? "))
            except:
                continue
            try:
                layers = int(input("Layers? "))
            except:
                continue
            print("\nNN is a basic Neural Network.")
            t = input("Enter nn or cancel: ").lower()
            if t == "nn":
                #ai = NeuralNetwork(mil * bpc, mol * bpc, bpc, [(mil * bpc, [])] * layers) 
                #TODO: fix this.
            else:
                continue
            print("Model created.")
            save(ai)
            break

# ai.setInOut(mil * bpc, mol * bpc)

# ai.bits_per_character = bpc


def train(amt=100, file="basic"):
    if amt <= 0:
        return
    data = loadTrainingData(file)
    print(f'Training with {amt} iterations on "{file}" dataset ({len(data[0])})...')
    try:
        if ai.type == "NN":
            ai.train(*data, amt)
        elif ai.type == "TCM":
            dataset = []
            for example in data:
                dataset.append({"prompt": example, "completion": data[example]})
            ai.train(dataset, amt)
        print("Training Complete")
    finally:
        save(ai)


register(save, ai)

# decode(ai.think([1, 0, 1, 0, 1]), bpc)
# ai.train(
#     [[1, 1, 1, 1]],
#     [[0, 0, 0, 0]],
#     2
# )

while True:
    inp = input(fg(70, 70, 255) + "You: " + rs.all)
    print(rs.all, end="")
    if inp.startswith("$help"):
        print(
            "Commands\n$exit\n$help\n$json-train [iterations=100] [file=basic]\n$text-train [iterations=1] [batch_size=200] [folder=data]"
        )
    elif inp.startswith("$exit"):
        break
    elif inp.startswith("$json-train"):
        x = inp.split(" ")[1:]
        if len(x) < 1 or x[0] == "":
            amt = 100
        else:
            amt = int(x[0])
        if len(x) < 2 or x[1] == "":
            f = "basic"
        else:
            f = x[1]
        train(amt, f)
        save(ai)
    elif inp.startswith("$text-train"):
        x = inp.split(" ")[1:]
        if len(x) < 1 or x[0] == "":
            iters = 1
        else:
            iters = int(x[0])
        if len(x) < 2 or x[1] == "":
            batch_size = 200
        else:
            batch_size = x[1]
        if len(x) < 2 or x[1] == "":
            folder = "./data"
        else:
            folder = x[1]
        ai.train(*dataloader.load_data(folder, mil, mol), iters, batch_size, 0.2, 1, 1)
        save(ai)
    else:
        resp = decode(ai.think(inp), bpc)
        print(
            f"{fg(255, 70, 70)}AI:{fg(255, 150, 150)} {resp[0]}\n{fg(255, 70, 70)}Confidence:{fg(255, 150, 150)} {resp[1]}%{rs.all}"
        )
