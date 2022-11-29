# Sandia_Project


The models of interest are in the Best_Model.ipynb file.
Run the notebook with the 'Strat_Temp_AOD_Flux.csv' file in the same directory.
The notebook trains two linear convolutional neural networks that take (16,48,24) tensors and output (16,48,6) tensors of temperatures
The 16x48 correspond to the spatial grid of latitudes and longitudes and the last dimension are the time steps
Use torch.save to save the model (documentation here: https://pytorch.org/docs/stable/generated/torch.save.html)
The two models trained use different training datasets (which can be changed as desired)
Training set A: temperatures between 1986-01-01 through 1990-07-01
Training set B: temperatures between 1986-01-01 through 1991-06-01
