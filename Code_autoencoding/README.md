# Over-parameterized Autoencoder Training Example

This codebase provides sample code to train the over-paramaterized autoencoder used in our paper on a toy dataset of 100 random samples.  The codebase additionally provides code to replicate the autoencoder from our paper on CMap provided that the user has downloaded the GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx dataset.  

## Dependencies

Python 3.7, PyTorch 1.6 (CUDA enabled), cmapPy 4.0.1

## Expected Output

The output consists of the number of data samples used for training and testing, the number of parameters in the model, and the training and test loss at each epoch.  The trained model will also be checkpointed every epoch for which the test loss improved.

## Instructions to Run Data

The following code will begin training the autoencoder:

python -u main.py

## Expected Runtime

Each epoch should take roughly .05 seconds when run on the GPU, but the CPU time should be comparable given that the network consists of only 1 hidden layer.  On the full CMap dataset, each epoch should take roughly 15 seconds on the GPU.  We set the epoch limit at 100000 and use a threshold of 1e-15 to terminate early in this toy example.  In the worst case the full 100000 epochs should take around 1.38 hours on this toy dataset. 
