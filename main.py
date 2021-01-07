import os

from flask import Flask, request
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.models import TextClassifier
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
app = Flask(__name__)

NEW_DATASETS_PATH = "../data/new"

def load_dataset_from(csv_file: str) -> pd.DataFrame:

    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        print("File couldn't be found. Verify if '{f_path}' is a correct file path!".format(f_path=csv_file))
        exit(1)


@app.route('/upload')
def index():
    mylist = ["siema", "banana", "cherry"]
    return mylist.__str__()

@app.route('/upload-file', methods=["POST"])
def upload_file():
    uploaded_file = request.files['file'];
    if uploaded_file.filename != '':
        ## uploaded_file jako parametr pandas
        ## cala logika tworzenia positive i negative
        df = load_dataset_from(uploaded_file)

        # To verify if data loaded correctly:
        # print(df.head(10))
        print(df.head(10))
        classifier = TextClassifier.load('sentiment')

        tmp_negatives = {}
        tmp_positives = {}

        print(f'Number of players: {len(df["Player"].unique())}')

        for player_name in df["Player"].unique():
            tmp_negatives[player_name] = list()
            tmp_positives[player_name] = list()

        for dp in df.values:
            l = dp.tolist()
            sentence = Sentence(l[-1])
            classifier.predict(sentence)
            if sentence.labels[0].value == "NEGATIVE":
                tmp_negatives[l[-2]].append(sentence.labels[0].score)
            elif sentence.labels[0].value == "POSITIVE":
                tmp_positives[l[-2]].append(sentence.labels[0].score)

        negative = {}
        positive = {}

        for player in tmp_negatives.keys():
            if len(tmp_negatives[player]) > 10:
                negative[player] = sum(tmp_negatives[player]) / len(tmp_negatives[player])

        for player in tmp_positives.keys():
            if len(tmp_positives[player]) > 10:
                positive[player] = sum(tmp_positives[player]) / len(tmp_positives[player])

        print(negative)
        print(positive)

        # for player in tmp.keys():
        #     tmp[player] = Sentence(tmp[player])
        #     classifier.predict(tmp[player])
        #     if tmp[player].labels[0].value == "NEGATIVE":
        #         negative[player] = tmp[player].labels[0].score
        #     elif tmp[player].labels[0].value == "POSITIVE":
        #         positive[player] = tmp[player].labels[0].score
        #


        positive_json = json.dumps(positive, indent=4)
        negative_json = json.dumps(negative, indent=4)
        list = {}
        list[0] = positive_json
        list[1] = negative_json
        return list

if __name__ == "__main__":
    app.run(debug=True)
