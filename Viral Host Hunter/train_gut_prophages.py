import pandas as pd
import utils
import pickle
import argparse
from Bio import SeqIO
from gut_prophages_info import info
from embedding import get_embedding
from encoder_only_dna import encoder
from dataset import MyDataset
from torch import optim, nn
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from autoencoder import AutoEncoder
from model import DnaPathNetworks
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from utils import *
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_protein', type=str, required=True)
    parser.add_argument('--train_dna', type=str, required=True)
    parser.add_argument('--val_protein', type=str, required=True)
    parser.add_argument('--val_dna', type=str, required=True)
    parser.add_argument('--level', choices=['family', 'genus', 'species'], type=str, help='Model classification level', required=True)
    parser.add_argument('--type', choices=['tail', 'lysin'], type=str, required=True)
    args = parser.parse_args()
    return args

utils.set_seed(config.seed)
args = parse()

train_protein = args.train_protein
train_dna = args.train_dna
val_protein = args.val_protein
val_dna = args.val_dna
type = args.type
level = args.level
embedding_file = f"./embedding/gut_prophages/{type}/{level}/"


train_cds = [str(rec.seq) for rec in SeqIO.parse(train_dna, 'fasta')]
train_proteins = [str(rec.seq) for rec in SeqIO.parse(train_protein, 'fasta')]
train_labels = [info[type][level][rec.description.split("#")[-1]] for rec in SeqIO.parse(train_protein, 'fasta')]
train_dataset = MyDataset(train_cds, train_labels, config.k)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
config.num_class = len(set(train_labels))

val_cds = [str(rec.seq) for rec in SeqIO.parse(val_dna, 'fasta')]
val_proteins = [str(rec.seq) for rec in SeqIO.parse(val_protein, 'fasta')]
val_labels = [info[type][level][rec.description.split("#")[-1]] for rec in SeqIO.parse(val_protein, 'fasta')]
val_dataset = MyDataset(val_cds, val_labels, config.k)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# generate embedding
embedding_path1 = embedding_file +  "train_embedding.csv"
embedding_path2 = embedding_file + "train_dna_embed.csv"
embedding_path3 = embedding_file +  "val_embedding.csv"
embedding_path4 = embedding_file + "val_dna_embed.csv"

if not os.path.exists(embedding_path1):
    train_seq = []
    for i in range(len(train_proteins)):
        zj = ''
        for j in range(len(train_proteins[i]) - 1):
            zj += train_proteins[i][j] + ' '
        zj += train_proteins[i][-1]
        train_seq.append(zj)
    train_emb = get_embedding(train_seq)
    np.savetxt(embedding_path1, train_emb, delimiter=',')

    train_feature = encoder.features(train_cds)
    np.savetxt(embedding_path2, train_feature, delimiter=',')

if not os.path.exists(embedding_path3):
    val_seq = []
    for i in range(len(val_proteins)):
        zj = ''
        for j in range(len(val_proteins[i]) - 1):
            zj += val_proteins[i][j] + ' '
        zj += val_proteins[i][-1]
        val_seq.append(zj)
    val_emb = get_embedding(val_seq)
    np.savetxt(embedding_path3, val_emb, delimiter=',')

    val_feature = encoder.features(val_cds)
    np.savetxt(embedding_path4, val_feature, delimiter=',')

standard_scaler = StandardScaler()
train_embedding = pd.read_csv(embedding_file + 'train_embedding.csv', header=None)
train_embedding = np.array(train_embedding)
train_dna_embed = pd.read_csv(embedding_file + 'train_dna_embed.csv', header=None)
train_dna_embed = np.array(train_dna_embed)
train_embedding = np.concatenate((train_embedding, train_dna_embed), axis=1)
standard_scaler = standard_scaler.fit(train_embedding)
utils.save_pickle(standard_scaler, f"./model/gut_prophages/{type}/{level}/standard_scaler.pkl")

val_embedding = pd.read_csv(embedding_file + 'val_embedding.csv', header=None)
val_embedding = np.array(val_embedding)
val_dna_embed = pd.read_csv(embedding_file + 'val_dna_embed.csv', header=None)
val_dna_embed = np.array(val_dna_embed)
val_embedding = np.concatenate((val_embedding, val_dna_embed), axis=1)

train_embedding = standard_scaler.transform(train_embedding)
val_embedding = standard_scaler.transform(val_embedding)

train_embedding = torch.from_numpy(train_embedding).float().to(config.device)
val_embedding = torch.from_numpy(val_embedding).float().to(config.device)


# initialize DHH
DHH = DnaPathNetworks().to(config.device)
optimizer = optim.Adam(DHH.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
loss_function = torch.nn.CrossEntropyLoss().to(config.device)

# train DHH
best_acc = float('-inf')
for epoch in range(1, config.epochs + 1):
    train_acc = utils.train(DHH, train_dataloader, train_embedding, loss_function, optimizer)
    scheduler.step()
    val_acc = utils.eval(DHH, val_dataloader, val_embedding)
    print('Epoch: %03d | Train ACC: %.4f | Val ACC: %.4f' % (epoch, train_acc, val_acc))
    if val_acc > best_acc:
        print('Val ACC improved, from %.4f to %.4f' % (best_acc, val_acc))
        best_acc = val_acc
        torch.save(DHH.state_dict(), f"./model/gut_prophages/{type}/{level}/model.pth")
    else:
        print('Val ACC did not improve from %.4f' % (best_acc))


# train autoencoder
DHH = DnaPathNetworks().to(config.device)
DHH.load_state_dict(torch.load(f"./model/gut_prophages/{type}/{level}/model.pth"))

train_dataloader_neat = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)

# feature extract
epoch_train_feature = feature_extract(DHH, train_dataloader_neat, train_embedding)
epoch_val_feature = feature_extract(DHH, val_dataloader, val_embedding)

train_feature_dataset = TensorDataset(epoch_train_feature)
train_feature_dataloader = DataLoader(train_feature_dataset, batch_size=config.batch_size, shuffle=True)
val_feature_dataset = TensorDataset(epoch_val_feature)
val_feature_dataloader = DataLoader(val_feature_dataset, batch_size=config.batch_size, shuffle=False)

# initialize autoencoder
input_dim = 1280 * 4 + 1024 + 133
hidden_dims = [4096, 2048, 1024]
autoencoder = AutoEncoder(input_dim, hidden_dims).to(config.device)
loss_function = torch.nn.MSELoss().to(config.device)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)

# train
best_val_loss = float('inf')
for epoch in range(1, 150 + 1):
    train_loss = autoencoder_train(autoencoder, train_feature_dataloader, loss_function, optimizer)
    val_loss = autoencoder_eval(autoencoder, val_feature_dataloader, loss_function)
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(autoencoder.state_dict(), f"./model/gut_prophages/{type}/{level}/autoencoder.pth")

# train RF
train_feature_dataset = TensorDataset(epoch_train_feature)
train_feature_dataloader = DataLoader(train_feature_dataset, batch_size=config.batch_size, shuffle=False)

autoencoder = AutoEncoder(input_dim, hidden_dims).to(config.device)
autoencoder.load_state_dict(torch.load(f"./model/gut_prophages/{type}/{level}/autoencoder.pth"))

X_train = autoencoder_encoder(autoencoder, train_feature_dataloader)
X_val = autoencoder_encoder(autoencoder, val_feature_dataloader)
y_train = np.array(train_labels)
y_val = np.array(val_labels)

# train
best_acc = float('-inf')
for min_samples_leaf in [1, 2, 3, 4]:
    for min_samples_split in [2, 3, 4]:
        for n_estimators in [50, 100, 150, 200]:
            clf = RandomForestClassifier(random_state=42, class_weight='balanced',
                                         max_features='sqrt',
                                         min_samples_leaf=min_samples_leaf,
                                         min_samples_split=min_samples_split,
                                         n_estimators=n_estimators,
                                         n_jobs=-1)
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            val_acc = accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                utils.save_pickle(clf, f"./model/gut_prophages/{type}/{level}/rf.pth")

print("Train Done!")
