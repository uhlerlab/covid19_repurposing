import argparse
from configs.blanco_configs import BLANCO_DATA_DIR, L1000_DATA_DIR

def setup_options():
    options = argparse.ArgumentParser()

    #options.add_argument('-d', action='store', dest='data',
    #                     default=L1000_DATA_DIR + 'GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx')
    return options.parse_args()
