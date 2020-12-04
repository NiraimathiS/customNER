#!/usr/bin/env python3

################################################################################
# Script to test custom named entity model using spaCy
# https://spacy.io/
################################################################################

import argparse
import json
import pandas as pd
import spacy
from spacy.scorer import Scorer
from spacy.gold import GoldParse
import sys


def get_test_data(testing_file, rows):
    """Function to load the test data
    """

    df_test = pd.read_excel(testing_file)

    # Remove rows that have NA
    if df_test.isnull().values.any():
        df_test.dropna(inplace=True)
        df_test.reset_index(drop=True, inplace=True)

    print("Using first {} rows for testing".format(rows))
    df_test = df_test[0:rows]

    return df_test


def test_model(model, df_test):
    """Function to test the loaded model
    """

    test_data = df_test['Question']
    print("Loading model '{}' to test ...".format(model))
    nlp = spacy.load(model)
    for text in test_data:
        print("Text: '{}'".format(text))
        doc = nlp(text)
        for ent in doc.ents:
            print("\t{} {} {} {}".format(ent.label_, ent.text, ent.start_char, ent.end_char))
    
    return nlp


def calculate_score(nlp, df_test):
    """Function to calculate the score/performance of the model
    """

    testing_dataset = []
    for index in range( 0, len(df_test.index) ):
        entities = []
        for parameter in json.loads(df_test['Parameters'][index]):
            entities.append(tuple(parameter))
        testing_dataset.append((df_test['Question'][index], {"entities" : entities}))

    # https://spacy.io/api/scorer#score
    scorer = Scorer()
    try:
        for input, annotation in testing_dataset:
            doc_gold_text = nlp.make_doc(input)
            correct_annot = GoldParse(doc_gold_text, entities=annotation['entities'])
            predicted_annot = nlp(input)
            scorer.score(predicted_annot, correct_annot)
            print(scorer.scores)
    except Exception as e:
        print(e)


def parse_arguments(args):
    """Function to parse input arguments
    """

    parser = argparse.ArgumentParser(description='Test NER')
    
    parser.add_argument(
        '--model',
        help='Path of model to be tested',
        required=True
        )
    parser.add_argument(
        '--test_data',
        help='Path of testing data',
        required=True
        )
    parser.add_argument(
        '--rows',
        help='Number of rows to be considered for testing',
        type=int,
        default=20
        )
    
    parsed_args = parser.parse_args(args)

    return parsed_args


if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    test_dataframe = get_test_data(args.test_data, args.rows)
    nlp = test_model(args.model, test_dataframe)
    calculate_score(nlp, test_dataframe)