import os
import re
import time
from json import load



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

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "dr.": "doctor",
    "mr.": "mister",
    "mrs.": "missus",
    "ms.": "miss",
    "jr.": "junior",
    "sr.": "senior",
    "co.": "company",
    "inc.": "incorporated",
    "ltd.": "limited",
    "a.i.": "artificial intelligence",
}
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
                if word in contractions:
                    word = contractions[word]
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
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet, stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from numpy import random, array, exp, dot, ndarray
    
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    stop_words = set(stopwords.words("english"))
    
    lemmatizer = WordNetLemmatizer()
    for filename in os.listdir("data"):
        print(filename)
        filepath = os.path.join("data", filename)
        with open(filepath, "r", encoding="utf-8") as f:
            with open(os.path.join("processed_data", filename), "w") as newf:
                new = preprocess_text(f.read().lower().replace("\t", "").replace('\n', ' '))
                new = re.sub('\s+', ' ', new)
                newf.write(new)
