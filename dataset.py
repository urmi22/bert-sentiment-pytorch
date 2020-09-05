import pdb
import torch
import config
import pandas as pd

from torch.utils.data import Dataset, DataLoader




class ImdbReviewDataset(Dataset):
    
    def __init__(self):
        self.datapath = config.DATA_PATH
        dataset = pd.read_csv(self.datapath).fillna("none")
        pdb.set_trace()


    def __len__(self):
        pass

    def __getitem__(self, index):
        pass




if __name__ == "__main__":
    imdb_dataset = ImdbReviewDataset()