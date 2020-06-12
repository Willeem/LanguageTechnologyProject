# What's in a machine, that's not in a human?

This is the repository for a Language Technology Project, a master's course at RUG university.

## Dependencies
The dependencies are easy to install with the provided requirements file. It is recommended to create a virtual environment and run the following code:
```
pip install -r requirements.txt
```

## Running the models
The baseline can be run with the following command:
```
python classifier_svm.py
```

The transformer models can be run in the following way:
```
python classifier_trf.py --arch <model_name> --num_epochs <int> (--test)
```
Where model\_name can be BERT, XLNet or RoBERTa, num\_epochs can be any integer. The --test argument should only be used if one wants to reproduce our test set results. Omit this argument to evaluate using the development set.

## Obtaining json files for the visualisation tool
The json files needed for the NeAt visualisation tool can be obtained by running the obtain\_attention\_scores.ipynb. Our results were obtained by using Google Colab. It requires a trained simpletransformers model in the same path as the .ipynb file.