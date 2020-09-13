import config
import numpy as np
import torch
import torch.nn as nn


from dataset import ImdbReviewDataset
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW
from model import BertBaseUncased
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def main():
    imdb_dataset = ImdbReviewDataset()
    dataset_length = len(imdb_dataset)
    indices = list(range(dataset_length))
    split = int(dataset_length * config.TRAIN_SPLIT)

    if config.SHUFFLE_DATASET:
        np.random.seed(config.RANDOM_SEED)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(imdb_dataset, batch_size = config.TRAIN_BATCH_SIZE, sampler = train_sampler)
    val_loader = torch.utils.data.DataLoader(imdb_dataset, batch_size = config.VAL_BATCH_SIZE, sampler = val_sampler)

    device = torch.device(config.DEVICE)
    model = BertBaseUncased()
    model.to(device)
    loss = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), config .LEARNING_RATE)

    for epoch in range(config.EPOCHS):

        per_epoch_loss = train_fn(train_loader, model, loss, optimizer, device)
        print("  epoch : {}, loss : {:.2f}".format(epoch, per_epoch_loss))
        val_fn(val_loader, model, loss, device)
      
    torch.save(model.state_dict(), config.MODEL_PATH)





def train_fn(data_loader, model, loss, optimizer, device):
    model.train()
    
    total_batch_loss = 0
    for bi, data_dict in tqdm(enumerate(data_loader), total = len(data_loader)):
        ids = data_dict["ids"].to(device, dtype = torch.long)
        token_type_ids = data_dict["token_type_ids"].to(device, dtype = torch.long)
        mask = data_dict["mask"].to(device, dtype = torch.long)
        targets = data_dict["targets"].to(device, dtype = torch.float)

        optimizer.zero_grad()
        outputs = model(ids = ids, mask = mask, token_type_ids = token_type_ids).squeeze(1)
        batch_loss = loss(outputs, targets)
        batch_loss.backward()
        optimizer.step()
        total_batch_loss = total_batch_loss + batch_loss

    return (total_batch_loss / len(data_loader))





def val_fn(data_loader, model, loss, device):
    model.eval()
    with torch.no_grad():
        ground_truth, pred = [], []
        total_batch_loss = 0
        for bi, data_dict in tqdm(enumerate(data_loader), total = len(data_loader)):
            ids = data_dict["ids"].to(device, dtype = torch.long)
            token_type_ids = data_dict["token_type_ids"].to(device, dtype = torch.long)
            mask = data_dict["mask"].to(device, dtype = torch.long)
            targets = data_dict["targets"].to(device, dtype = torch.float)

            outputs = model(ids = ids, mask = mask, token_type_ids = token_type_ids).squeeze(1)
            batch_loss = loss(outputs, targets)
            total_batch_loss = total_batch_loss + batch_loss.item()
            targets = list(targets.cpu().detach().numpy())
            ground_truth.append(targets)
            outputs = [1.0 if item >= 0.5 else 0.0 for item in outputs]
            pred.append(outputs)
            

        
        avg_loss = total_batch_loss / len(data_loader)
        print("Average validation loss: {:.2f}".format(avg_loss))
        flat_ground_truth = [item for sublist in ground_truth for item in sublist]
        flat_pred = [item for sublist in pred for item in sublist]
        c_m = confusion_matrix(flat_ground_truth, flat_pred)
        accuracy = 100 * (np.array(flat_ground_truth) == np.array(flat_pred)).sum() / len(flat_ground_truth)
        print("Confusion matrix : {}".format(c_m))
        print("Accuracy : {:.2f}%".format(accuracy))
        







if __name__ == "__main__":
    main()