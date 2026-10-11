# flightBufferModel

parquet file source:
https://www.kaggle.com/datasets/dianatofficial/us-flight-on-time-performance-2010-2026

Download bts_2026.parquet and place in data/raw


# Workflow files
Workflow generally goes as follows:
1. (data_cleaning.py) 
    a.parquet file goes in during data_cleaning.py and is converted into a dataframe (pandas table-like structure) with columns of interest and invalid rows trimmed

2. (features.py)
    a. That dataframe then goes into features.py where rows are divided into categorical input, numerical input, and target output.
    b. Numerical data is analyzed for means and standard deviations, categorical input is mapped to numerical values for analysis.
    c. Then numerical data is z-scored and categorical data is encoded with an extra bucket for miscellaneous categorcial data.
    d. Dataframe including numerical z-scores, categorical data encoded, and encoder keys are outputted. 
    e. There is a seperate function to get the length of our encodeers(how many encoders we have in each category.)

3. (dataset.py)
    a. Defines our pytorch dataset class using a pandas dataframe.
    b. cat num and target are all stored as tensors (a multidimensional array that is differentiable and GPU-capable)

4. (model.py)
    a. This is our representation of a neural network. It will broadly take in data and output model architecture for model.pt(pytorch file for later use)
    b. The following methods are used in this order to create our network:
        b1. .Linear(a,b) - takes in input a numbers and produces b numbers using weights. input_dim is input dimensions (categorical and numerical columns)
        b2. .ReLU() - introduces non-linearity by changing negative values to zero. This allows our data to not just be one big line and learn more complex patterns.
        b3. .dropout(.2) - ignores 20% of our neurons. This helps us avoid overfitting
    c. Data is Linearized, relu'd, dropped, linearized, relu'd, and finally linearized into a final number
    d. forward function pushes a singular flight through the mlp (multi layer perceptron)

5. (train.py)
    a. This gets our data and writes results to our artifacts.
    b. Data is first loaded, then split 80/20 where 20 is reserved for training validation.
    c. Two dataframes are then sent through features.py: training and validation frames.
    d. The dataframes are turned into datasets using dataset.py
    e. A Delaynet is constructed using model.py
    f. Training loop runs: Grabs a batch, runs through network, compares to actual data, adjusts which weights are determined to be at fault. (aka. backpropogation and gradient descent)
    g. Writes results to artifacts.

6. (calibrate_buffer.py)
    a. This gets us a buffer so that people can withstand bad days. By all means this could be calibrated based off of risk/if people want shorter connections.
    b. First it rebuilds the model based off of metadata.
    c. Then it collects a batch of predictions, compares our model's outup based off of the actual values, and then builds a 90th percentile on the late side for a buffer.
    d. Writes the buffer to metadata for later use.

7. (predict.py)
    a. Brings in metadata, model, and encoders from artifacts and reconstructs the model with designated weights.
    b. encode_row encodes a flight into the tensor that the model will work with.
    c. predict returns the one liner prediction while predict_buffer returns 90th percentile prediction data

# Artifacts
encoders.pkl - stores encoder data
metadata.json - stores data on model dimensions, statistics, and values
model.pt - the pytorch model file used for predictions
