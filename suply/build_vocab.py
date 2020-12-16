import pickle, json, os, re, pickle, argparse
from collections import Counter
from tqdm import *
import numpy as np

# vacabulary class
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {} # {word: index}
        self.idx2word = {} # {index: word}
        self.idx = 0 

    def add_word(self, word, idx=None):
        if idx is None:
            if not word in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
            return self.idx
        else:
            if not word in self.word2idx:
                self.word2idx[word] = idx
                if idx in self.idx2word.keys():
                    self.idx2word[idx].append(word)
                else:
                    self.idx2word[idx] = [word]

                return idx

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<pad>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

# read words from the txt
def get_ingredients(line):
    line = line.lower()
    items = line.split(",")
    result = []
    for i in range(len(items)):
        if len(items[i]) > 0:
            items[i] = items[i].replace(' ', '_')
            items[i] = items[i].strip('\n')
            result.append(items[i])
    return result
    

# for preprocess ingredients words
# cluster similer words
def cluster_ingredients(counter_ingrs):
    mydict = dict()
    mydict_ingrs = dict()

    for k, v in counter_ingrs.items():

        w1 = k.split('_')[-1]
        w2 = k.split('_')[0]
        lw = [w1, w2]
        if len(k.split('_')) > 1:
            w3 = k.split('_')[0] + '_' + k.split('_')[1]
            w4 = k.split('_')[-2] + '_' + k.split('_')[-1]

            lw = [w1, w2, w4, w3]

        gotit = 0
        for w in lw:
            if w in counter_ingrs.keys():
                # check if its parts are
                parts = w.split('_')
                if len(parts) > 0:
                    if parts[0] in counter_ingrs.keys():
                        w = parts[0]
                    elif parts[1] in counter_ingrs.keys():
                        w = parts[1]
                if w in mydict.keys():
                    mydict[w] += v
                    mydict_ingrs[w].append(k)
                else:
                    mydict[w] = v
                    mydict_ingrs[w] = [k]
                gotit = 1
                break
        if gotit == 0:
            mydict[k] = v
            mydict_ingrs[k] = [k]

    return mydict, mydict_ingrs


# for preprocess ingredients words
# remove plurals
def remove_plurals(counter_ingrs, ingr_clusters):
    del_ingrs = []

    for k, v in counter_ingrs.items():

        if len(k) == 0:
            del_ingrs.append(k)
            continue

        gotit = 0
        if k[-2:] == 'es':
            if k[:-2] in counter_ingrs.keys():
                counter_ingrs[k[:-2]] += v
                ingr_clusters[k[:-2]].extend(ingr_clusters[k])
                del_ingrs.append(k)
                gotit = 1

        if k[-1] == 's' and gotit == 0:
            if k[:-1] in counter_ingrs.keys():
                counter_ingrs[k[:-1]] += v
                ingr_clusters[k[:-1]].extend(ingr_clusters[k])
                del_ingrs.append(k)

    for item in del_ingrs:
        del counter_ingrs[item]
        del ingr_clusters[item]

    return counter_ingrs, ingr_clusters

# build_vocabl_dataset
def build_vocab_dataset():
    
    f1 = open(os.getcwd() +"/data/Ingredients101/Annotations/classes.txt", "r")
    idx = 0
    class_names = {}
    for class_name in f1:
        class_name = class_name.strip()
        class_names[idx] = class_name
        idx += 1
    f1.close()
    
    f = open(os.getcwd() +"/data/Ingredients101/Annotations/ingredients.txt", "r")
    class_to_ingrd = {}
    counter_ingrs = Counter()
    idx = 0
    for line in f:
        items = get_ingredients(line)
        class_to_ingrd[class_names[idx]] = items
        idx += 1
        counter_ingrs.update(items)
    
    f.close()
    counter_ingrs, cluster_ingrs = cluster_ingredients(counter_ingrs)
    counter_ingrs, cluster_ingrs = remove_plurals(counter_ingrs, cluster_ingrs)
    vocab_ingrs = Vocabulary()
    idx = vocab_ingrs.add_word('<end>')
    for k, _ in counter_ingrs.items():
        for ingr in cluster_ingrs[k]:
            idx = vocab_ingrs.add_word(ingr, idx)
        idx += 1
    _ = vocab_ingrs.add_word('<pad>', idx)
    
    print("Total ingr vocabulary size: {}".format(len(vocab_ingrs)))
    dataset = {'train': [], 'val': [], 'test': []}
    for split in ['train', 'val', 'test']:
        f2 = open(os.getcwd() +"/data/Ingredients101/Annotations/"+split+'_images.txt', "r")
        temp_name = None
        idx = -1
        for line in f2:
            #print(os.getcwd() + "/data/food-101/food-101/sampled_images/" + line + ".jpg")
            line = line.strip()
            if os.path.exists( os.getcwd() + "/data/food-101/food-101/sampled_images/" + line +".jpg"):
                name, number = line.split('/')
                if name != temp_name:
                    idx += 1
                    temp_name = name
                newentry = {'id': temp_name,'ingredients': class_to_ingrd[class_names[idx]], 'images': line}
                dataset[split].append(newentry)
    print(len(dataset['train']), len(dataset['test']), len(dataset['val']))
    return vocab_ingrs, dataset


def main():
    ingr_path = os.getcwd() + "/data/Recipes5k/annotations/classes_Recipes5k.txt"
    anno_path = os.getcwd() + "/data/Recipes5k/annotations/"
    vocab_ingrs, dataset = build_vocab_dataset(ingr_path, anno_path)

    with open(os.path.join(os.getcwd(), 'vocab_ingrs.pkl'), 'wb') as f:
        pickle.dump(vocab_ingrs, f)

    for split in dataset.keys():
        with open(os.path.join(os.getcwd(),'data' + split + '.pkl'), 'wb') as f:
            pickle.dump(dataset[split], f)





