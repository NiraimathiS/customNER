#!/usr/bin/env python3

################################################################################
# Script to create and train a custom named entity model using spaCy
# https://spacy.io/
################################################################################

import argparse
import json
import os
import pandas as pd
from pathlib import Path
import random
import spacy
import sys
from spacy.util import minibatch, compounding
import time


def prepare_training_data(training_file, rows):
    """Funtion to load the training data and to clean it
    Converts the data to the below format:
    (
        "INPUT TEXT",
        {"entities": [(start_index, end_index, LABEL)]},
    )
    """

    if not os.path.isfile(training_file):
        raise FileNotFoundError("Training data set file '%s' not found!" % (training_file))

    df_train = pd.read_excel(training_file)

    # Remove rows that have NA
    if df_train.isnull().values.any():
        df_train.dropna(inplace=True)
        df_train.reset_index(drop=True, inplace=True)

    print("Using first {} rows for training".format(rows))
    df_train = df_train[0:rows]

    training_dataset = []
    for index in range( 0, len(df_train.index) ):
        entities = []
        for parameter in json.loads(df_train['Parameters'][index]):
            entities.append(tuple(parameter))
        training_dataset.append((df_train['Question'][index], {"entities" : entities}))

    return training_dataset


def get_named_entities(training_dataset):
    """Function to get the list of named entities to be trained
    """

    named_entities = []

    for _, annotations in training_dataset:
        for ent in annotations.get("entities"):
            if ent[2] not in named_entities:
                named_entities.append(ent[2])

    named_entities.sort()

    return named_entities


def train_model(lang_cls, load_model, training_dataset, train_iter):
    """Function to train NER
    """

    # load model if provided or else create a blank model for the given language class
    if load_model is not None:
        nlp = spacy.load(load_model)
        print("Loading provided model '{}' ...".format(load_model))
    else:
        nlp = spacy.blank(lang_cls)
        print("Creating a blank '{}' model ...".format(lang_cls))

    # If 'ner' is not in the pipeline, add entity recognizer to model
    # https://spacy.io/usage/processing-pipelines
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    # From provided dataset identify and add new labels to entity recognizer
    named_entities = get_named_entities(training_dataset)
    for named_entity in named_entities:
        print("Adding custom label '{}' to entity recognizer".format(named_entity))
        ner.add_label(named_entity)

    # Inititalize optimizer
    print("Initializing optimizer ...")
    if load_model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    training_start_time = time.time()
    print("Training NER model ...")
    print("Number of training iterations: {}".format(train_iter))
    
    # Disable all pipes except 'ner' for training
    # https://spacy.io/usage/training
    disable_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*disable_pipes):
        for train_iter in range(10):
            random.shuffle(training_dataset)
            losses = {}
            batches = minibatch(training_dataset, size=compounding(1, 16, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print('Losses', losses)

    print("Training done")
    
    elapsed_time =  (time.time() - training_start_time)
    print("Training time",time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    
    return nlp


def save_model(nlp, output_dir):
    """Function to save the model to disk
    """

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Model saved to folder '{}'".format(output_dir))


def parse_arguments(args):
    """Function to parse input arguments
    """

    parser = argparse.ArgumentParser(description='Train NER')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--blank_model',
        help='Create a blank model of a given language class for training'
        )
    group.add_argument(
        '--load_model',
        help='Path of model to be trained'
        )
    parser.add_argument(
        '--train_data',
        help='Path to training data',
        required=True
        )
    parser.add_argument(
        '--rows',
        help='Number of rows to be considered for training',
        type=int,
        default=20
        )
    parser.add_argument(
        '--train_iter',
        help='Number of iterarions to train the model',
        type=int,
        default=10
        )
    parser.add_argument(
        '--save_model',
        help='Output directory to save the model'
        )


    parsed_args = parser.parse_args(args)

    return parsed_args


if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])
    training_dateset = prepare_training_data(args.train_data, args.rows)
    nlp = train_model(args.blank_model, args.load_model, training_dateset, args.train_iter)
    if args.save_model != None:
        save_model(nlp, args.save_model)

