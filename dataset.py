import pdb
import torch
import config
import pandas as pd

from torch.utils.data import Dataset, DataLoader
<<<<<<< HEAD


=======




>>>>>>> 43bd97585d281f5ca6c07e797fa2e8f3f5ef01e1
class ImdbReviewDataset(Dataset):
    
    def __init__(self):
        self.datapath = config.DATA_PATH
        self.dataset = pd.read_csv(self.datapath).fillna("none")
        self.reviews = [review for review in self.dataset['review']] # or self.dataset.review same as self.dataset['review']
        self.targets = [1 if sentiment == 'positive' else 0 for sentiment in self.dataset['sentiment']]
        self.tokenizer = config.TOKENIZER
        self.max_token_len = config.MAX_TOKEN_LEN
        


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        review = self.reviews[index]
        target = self.targets[index]

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens = True,
            max_length = self.max_token_len,
            padding = 'max_length',
            truncation = True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype = torch.long),
            "mask": torch.tensor(mask, dtype = torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype = torch.long),
            "targets": torch.tensor(target, dtype = torch.float)
        }


# if __name__ == "__main__":
    # imdb_dataset = ImdbReviewDataset()
    # dataset_len = len(imdb_dataset)
    # item_dict = imdb_dataset.__getitem__(0)
