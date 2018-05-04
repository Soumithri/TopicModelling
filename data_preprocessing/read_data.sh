#!/usr/bin/env bash

COLLECTION_NAME="historical_tweets2"
UNIQUE_USERS="unique_users_list.csv"
OUTPUT_DIR="data_preprocessed"

python data_preprocessing/read_data.py $COLLECTION_NAME $UNIQUE_USERS $OUTPUT_DIR
