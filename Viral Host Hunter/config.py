import torch


class Config:
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.epochs = 100
        self.batch_size = 64
        self.learning_rate = 0.001
        self.k = 4
        self.seed = 0



config = Config()