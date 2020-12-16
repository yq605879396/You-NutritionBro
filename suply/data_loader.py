import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os, pickle,random, json
import numpy as np

from PIL import Image
from build_vocab import Vocabulary


class Food101Dataset(data.Dataset):

    def __init__(self, data_dir, split, maxnumlabels, 
                 transform=None, max_num_samples=-1):
        
        # read vocabularies
        self.ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'vocab_ingrs.pkl'), 'rb'))
        
        # read data set - train/val/test
        self.dataset = pickle.load(open(os.path.join(data_dir, 'data'+split+'.pkl'), 'rb'))

        self.label2word = self.get_ingrs_vocab() # words or min(w, key= len)
        self.ids = [] # the index of entry in dataset
        self.split = split # train or val or test
        
        for i, entry in enumerate(self.dataset):
            if len(entry['ingredients']) == 0:
                continue
            self.ids.append(i)

        self.transform = transform 
        self.max_num_labels = maxnumlabels
        if max_num_samples != -1:
            random.shuffle(self.ids)
            self.ids = self.ids[:max_num_samples]

    def get_ingrs_vocab(self):
        return [min(w, key=len) if not isinstance(w, str) else w for w in
                self.ingrs_vocab.idx2word.values()]  # includes 'pad' ingredient

    def get_ingrs_vocab_size(self):
        return len(self.ingrs_vocab)
    

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        data_entry = self.dataset[self.ids[index]] # index th data
        img_id = data_entry['images'].strip()
        #print(img_id)
        
        #img_path = "./data/food-101/food-101/sampled_images/" + img_id + ".jpg" #os.path.join('./images/',img_id,".jpg")
        #print(img_path)
        img_path = "./data/Recipes5k/images/"+img_id+ ".jpg" 
        labels = self.dataset[self.ids[index]]['ingredients'] # ingredient list
        ilabels_gt = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>') # <pad><pad>...<pad><pad>
        
        pos = 0
        true_ingr_idxs = []    # add index of each word in integredient list
        
        for i in range(len(labels)):
            true_ingr_idxs.append(self.ingrs_vocab(labels[i]))

        for i in range(self.max_num_labels):
            if i >= len(labels):
                label = '<pad>'
            else:
                label = labels[i]
            label_idx = self.ingrs_vocab(label) # get word index
            if label_idx not in ilabels_gt: # not pad
                ilabels_gt[pos] = label_idx
                pos += 1

        ilabels_gt[pos] = self.ingrs_vocab('<end>') # add "end"
        ingrs_gt = torch.from_numpy(ilabels_gt).long()


        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image) # crop image
        image_input = image

    
        return image_input, ingrs_gt, img_id, img_path

    def __len__(self):
        return len(self.ids)


def collate_fn(data):


    image_input, ingrs_gt, img_id, path = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    image_input = torch.stack(image_input, 0)
    ingrs_gt = torch.stack(ingrs_gt, 0)

    return image_input, ingrs_gt, img_id, path


def get_loader(data_dir, split, 
               maxnumlabels, transform, batch_size,
               shuffle, num_workers, drop_last=False,
               max_num_samples=-1,
               ):

    dataset = Food101Dataset(data_dir=data_dir, split=split,
                              maxnumlabels=maxnumlabels,
                              transform=transform,
                              max_num_samples=max_num_samples,
                              )

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                              drop_last=drop_last, collate_fn=collate_fn, pin_memory=True)
    return data_loader, dataset


