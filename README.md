# Custom Entity Recognition using spaCy

Named Entity Recognition is the task of classifying a named entity present in a
piece of text into pre-defined categories like person, country, place,
organization, time etc

A model trained on classic literature cannot be used to predict names of diseases
present in a medical report. Model's preformance depends on training data.
This means we need a custom model that is good in predicting name of disease
from a medical report. This can be achieved by training the model with good
amount of annotated medical report. Using spaCy, it is possible to create a
custom NER from scratch based on a blank model.

## Prerequisites

The following python packages are used in the train and test ner scripts:

* spaCy - For modelling NER

* pandas - For data preprosessing and cleaning

## Usage

### To train the model

```bash
 python3 ner_train.py        [-h]
                             (--blank_model BLANK_MODEL | --load_model LOAD_MODEL)
                             --train_data TRAIN_DATA [--rows ROWS]
                             [--train_iter TRAIN_ITER]
                             [--save_model SAVE_MODEL]

Train NER

optional arguments:
  -h, --help                       show this help message and exit
  --blank_model BLANK_MODEL        Create a blank model of a given language
                                   class for training
  --load_model LOAD_MODEL          Path of model to be trained
  --train_data TRAIN_DATA          Path to training data
  --rows ROWS                      Number of rows to be considered for training
                                   Defaults to 20 rows
  --train_iter TRAIN_ITER          Number of iterarions to train the model
                                   Default is 10 iterations
  --save_model SAVE_MODEL          Optional: Output directory to save the model
```

#### Example

* Train a blank english model.
For details of language supported by spaCy, refer <https://spacy.io/usage/models#languages>

```bash
python3 ner_train.py --blank_model en --train_data ner_dataset_train.xlsx \
--rows 2000 --save_model my_model
```

* Train an already available model

```bash
python3 ner_train.py --load_model my_model --train_data ner_dataset_train.xlsx

```

### To test the model

```bash
python3 ner_test.py [-h] --model MODEL --test_data TEST_DATA [--rows ROWS]

Test NER

optional arguments:
  -h, --help               show this help message and exit
  --model MODEL            Path of model to be tested
  --test_data TEST_DATA    Path of testing data
  --rows ROWS              Number of rows to be considered for testing
                           Defaults to 20 rows
```

#### Example

Test the provided model

```bash
python3 ner_test.py --model my_model --test_data ner_dataset_test.xlsx --rows 5
```
