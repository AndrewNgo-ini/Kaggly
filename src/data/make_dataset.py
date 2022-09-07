# -*- coding: utf-8 -*-
# import click
# import logging
# from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer
from datasets import Dataset
# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')



def prepare_data_and_label(category):
    df = pd.read_csv("data/raw/train.csv")
    def label_map(x):
        mp = {
        1.0: 0,
        1.5: 1,
        2.0: 2,
        2.5: 3,
        3.0: 4,
        3.5: 5,
        4.0: 6,
        4.5: 7,
        5.0: 8,
        }
        return mp[x]
    df[category] = df.apply(lambda x: label_map(x[category]), axis=1)
    df = df.rename(columns={category:'labels'})
    #print(df.head(3))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def tokenize_fn(examples):
        return tokenizer(examples["full_text"], truncation=True)

    ds = Dataset.from_pandas(df.iloc[:3,:])
    ds = ds.map(tokenize_fn, batched=True, remove_columns=["full_text"])
    ds.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
    #ds.format['type']
    #ds = prepare_data(df)
    return ds

if __name__ == '__main__':
    print(prepare_data_and_label("cohesion"))


    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # # find .env automagically by walking up directories until it's found, then
    # # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    # main()
