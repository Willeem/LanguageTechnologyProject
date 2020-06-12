# What's in a machine, that's not in a human?

This is the repository for a study as part of Language Technology Project, a master's course at the University of Groningen. This study aims to employ Transformers to automatically distinguish human and machine translations. These are implemented using [SimpleTransformers](https://simpletransformers.ai).

## Dependencies
The dependencies are easy to install with the provided requirements file. It is recommended to create a virtual environment and run the following command:
```
pip install -r requirements.txt
```

## Running the models
The baseline SVM can be run with the following command:
```
python classifier_svm.py
```

The Transformer models can be run in the following way:
```
python classifier_trf.py --arch <model_name> --num_epochs <int> [--test]
```
Where `--model_name` can be `BERT`, `XLNet` or `RoBERTa`, `--num_epochs` can be any integer. The `--test` argument should only be used if one wants to reproduce the test set results. Omit this argument to evaluate using the development set.

## Obtaining JSON-files for visualisation
The JSON-files needed for the [_NeAt_ visualisation tool](https://cbaziotis.github.io/neat-vision/) can be obtained by running `obtain_attention_scores.ipynb`. The results were obtained by using Google Colab. It requires a trained [SimpleTransformers](https://simpletransformers.ai) model in the same path as the `.ipynb` file.
