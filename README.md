# Classify Questions
Trains and tests classification models on the QANTA question dataset. Models are trained to predict whether a given question is either college difficulty or high school difficulty.

## Usage
1. ```pip install -r requirements.txt```
2. ```python setup.py```
3. ```python run.py [classifiers to train and test]```
    ### Models included
    * BERT (`bert`)
    * DistilBERT (`distil-bert`)
    * ConvBERT (`conv-bert`)
    * ELECTRA (`electra`)

    Run with ```--test-only``` flag if models are already trained