import pdb
import torch
import config
import pandas as pd

from torch.utils.data import Dataset, DataLoader




class ImdbReviewDataset(Dataset):
    
    def __init__(self):
        self.datapath = config.DATA_PATH
        self.dataset = pd.read_csv(self.datapath).fillna("none")
        self.reviews = [review for review in self.dataset['review']] # or self.dataset.review same as self.dataset['review']
        self.targets = [1 if sentiment == 'positive' else 0 for sentiment in self.dataset['sentiment']]
        


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pass




if __name__ == "__main__":
    imdb_dataset = ImdbReviewDataset()
    dataset_len = len(imdb_dataset)
    pdb.set_trace()