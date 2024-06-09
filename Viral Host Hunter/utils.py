import os
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from config import config
from sklearn.metrics import accuracy_score


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, dataloader, embedding, loss_function, optimizer):
    epoch_labels, epoch_preds = [], []
    model.train()
    for index, x, y in tqdm(dataloader):
        optimizer.zero_grad()
        embed = embedding[index]
        x, y = x.to(config.device), y.to(config.device)
        preds = model(x, embed)
        batch_loss = loss_function(preds, y)
        batch_loss.backward()
        optimizer.step()
        with torch.no_grad():
            preds = torch.softmax(preds, dim=1)
            epoch_labels += list(y.cpu().numpy())
            epoch_preds += list(preds.argmax(1).cpu().numpy())
    accuracy = accuracy_score(epoch_labels, epoch_preds)
    return accuracy


def eval(model, dataloader, embedding):
    epoch_labels, epoch_preds, epoch_prob = [], [], []
    model.eval()
    with torch.no_grad():
        for index, x, y in dataloader:
            x, y = x.to(config.device), y.to(config.device)
            embed = embedding[index]
            preds = model(x, embed)
            preds = torch.softmax(preds, dim=1)
            epoch_prob += list(preds.cpu().numpy())
            epoch_labels += list(y.cpu().numpy())
            epoch_preds += list(preds.argmax(1).cpu().numpy())
        accuracy = accuracy_score(epoch_labels, epoch_preds)
    return accuracy

def test(model, dataloader, embedding):
    epoch_labels, epoch_preds, epoch_prob = [], [], []
    model.eval()
    with torch.no_grad():
        for index, x, y in dataloader:
            x, y = x.to(config.device), y.to(config.device)
            p_embed = embedding[index]
            preds = model(x, p_embed)
            preds = torch.softmax(preds, dim=1)
            epoch_prob += list(preds.cpu().numpy())
            epoch_preds += list(preds.argmax(1).cpu().numpy())
    return epoch_labels, epoch_preds, epoch_prob

def autoencoder_train(model, dataloader, loss_function, optimizer):
    epoch_train_loss = []
    model.train()
    for x in dataloader:
        optimizer.zero_grad()
        x = x[0].to(config.device)
        preds = model(x)
        batch_loss = loss_function(preds, x)
        batch_loss.backward()
        optimizer.step()
        epoch_train_loss.append(batch_loss.item())
    train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
    return train_loss

def autoencoder_eval(model, dataloader, loss_function):
    epoch_val_loss = []
    model.eval()
    with torch.no_grad():
        for x in dataloader:
            x = x[0].to(config.device)
            preds = model(x)
            batch_loss = loss_function(preds, x)
            epoch_val_loss.append(batch_loss.item())
        val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
    return val_loss

def feature_extract(model, dataloader, embedding):
    epoch_feature = []
    model.eval()
    with torch.no_grad():
        for index, x, y in dataloader:
            feature2 = embedding[index]
            x, y = x.to(config.device), y.to(config.device)
            feature = model.feature(x, feature2)
            epoch_feature += list(feature.detach().cpu().numpy())
    epoch_feature_array = np.array(epoch_feature)
    epoch_feature = torch.tensor(epoch_feature_array, dtype=torch.float32)
    return epoch_feature


def autoencoder_encoder(autoencoder, dataloader):
    epoch_feature = []
    autoencoder.eval()
    with torch.no_grad():
        for x in dataloader:
            x = x[0].to(config.device)
            feature = autoencoder.encoder(x)
            epoch_feature += list(feature.detach().cpu().numpy())
    return epoch_feature


def save_pickle(target, path):
    f = open(path, 'wb')
    pickle.dump(target, f)
    f.close()


def load_pickle(path):
    f = open(path, 'rb')
    target = pickle.load(f)
    f.close()
    return target


def int2str(infomation):
    dic = {v: k for k, v in infomation.items()}
    return dic