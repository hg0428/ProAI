import os
import re
import time
# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet, stopwords
# from nltk.tokenize import word_tokenize, sent_tokenize
from numpy import random, array, exp, dot, ndarray
from json import load

# nltk.download("averaged_perceptron_tagger", quiet=True)
# nltk.download("wordnet", quiet=True)
# nltk.download("stopwords", quiet=True)
# nltk.download("punkt", quiet=True)
# stop_words = set(stopwords.words("english"))

# lemmatizer = WordNetLemmatizer()


def fill(l, length, null=0, reverse=False):
    if len(l) > length:
        if reverse:
            return l[len(l) - length :]
        else:
            return l[:length]
    else:
        for x in range(length - len(l)):
            if reverse:
                l = [null] + l
            else:
                if isinstance(l, str):
                    l += str(null)
                else:
                    l.append(null)
    return l


def process_value(x, bits_per_character=8, tokenizer_funct=ord):
    if type(x) == str:
        return [
            int(i)
            for i in "".join(
                [format(tokenizer_funct(i), f"0{bits_per_character}b") for i in x]
            )
        ]
    elif type(x) == int:
        return [int(i) for i in format(x, f"0{bits_per_character}b")]
    elif type(x) == list:
        return x
    elif type(x) == array:
        return x
    elif type(x) == ndarray:
        return x


def decode(data, bits_per_character=8, decoder=chr):
    confidence = 0
    for i in data:
        if i < 0.5:
            confidence += 1 - i
        else:
            confidence += i
    confidence /= len(data)
    out = [round(x) for x in data]
    bytes = [
        out[x : x + bits_per_character] for x in range(0, len(out), bits_per_character)
    ]
    strbytes = ["".join([str(i) for i in x]) for x in bytes]
    chrs = [int(x, 2) for x in strbytes]
    string = ""
    for x in chrs:
        if x == 0:
            string += ""
        try:
            string += decoder(x)
        except:
            string += ""
    return string, confidence


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def preprocess_text(text):
    sentances = sent_tokenize(text)
    words = []
    newtext = ""
    for sentance in sentances:
        # Perform lemmatization
        for word in nltk.pos_tag(word_tokenize(sentance)):
            if word[1] == ".":
                if len(newtext) > 0 and newtext[-1] == " ":
                    newtext = newtext[:-1]
                if word[0] != "\n":
                    newtext += word[0]
            else:
                pos = get_wordnet_pos(word[1])
                if pos:
                    word = lemmatizer.lemmatize(word[0], pos)
                else:
                    word = word[0]
                if len(newtext) != 0 and newtext[-1] != " ":
                    newtext += " "
                newtext += word
    # Join the words back into a string
    return text.strip(" ")


def load_data(directory, max_in, max_out, bits_per_character=8, pre_proccessor=None):
    inputs = []
    outputs = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            if pre_proccessor:
                text = pre_proccessor(text)
            sentences = re.split("(?<=[.!?])\s+", text)
            combined_text = ""
            for sentence in sentences:
                if len(combined_text) + len(sentence) <= max_in + max_out:
                    combined_text += " " + sentence
                else:
                    while len(combined_text) > max_in:
                        inputs.append(combined_text[:max_in])
                        outputs.append(combined_text[max_in : max_in + max_out])
                        combined_text = combined_text[max_in:]
                    combined_text += sentence
            if len(combined_text) > 0:
                inputs.append(combined_text[:max_in])
                outputs.append(combined_text[max_in:])
        print(f"🔥 {filename}")
    # print(len(inputs), len(outputs), inputs[0])
    # [fill(process_value(inp, bits_per_character), max_in) for inp in inputs], [
    #    fill(process_value(out, bits_per_character), max_out) for out in outputs
    # ]
    return inputs, outputs


def loadJsonData(x, bits_per_character, max_in, max_out, pre_processor=lambda x: x):
    with open(f"training_data/{x}.json") as f:
        data = load(f)
    return [
        fill(process_value(pre_processor(inp), bits_per_character), max_in)
        for inp in data.keys()
    ], [
        fill(process_value(pre_processor(out), bits_per_character), max_out)
        for out in data.values()
    ]


if __name__ == "__main__":
    for filename in os.listdir("data"):
        filepath = os.path.join("data", filename)
        with open(filepath, "r", encoding="utf-8") as f:
            with open(os.path.join("processed_data", filename), "w") as newf:
                newf.write(preprocess_text(f.read().lower().replace("\t", "")))
