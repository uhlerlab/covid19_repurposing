import sys
sys.path.insert(0, '../')
import options_parser as op
import data_loader as dl
import numpy as np
import torch
import random
import trainer


# Trains an autoencoder on gene expression vectors from l1000
def main(args):
    SEED = 17
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    #train_loader, test_loader = dl.get_data(args.data)
    train_loader, test_loader = dl.get_data("")
    trainer.train_network(train_loader, test_loader)

if __name__ == "__main__":
    args = op.setup_options()
    main(args)
