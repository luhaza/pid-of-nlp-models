import pandas as pd
from sklearn.utils import shuffle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
full_file = os.path.join(current_dir, "IBC", "ibc.csv")
sample_file = os.path.join(current_dir, "IBC", "sample_ibc.csv")

ibc = pd.read_csv(full_file)
sample = pd.read_csv(sample_file)

dsq = sample["sentence"].to_list()
ft_ibc = ibc.loc[~ibc["sentence"].isin(dsq), :].copy()

ft_ibc.drop_duplicates(subset="sentence", inplace=True)
sample.drop_duplicates(subset="sentence", inplace=True)

ft_ibc = shuffle(ft_ibc, random_state=1)

conservative = ft_ibc.loc[ft_ibc['label'] == "Conservative"].head(1250)
liberal = ft_ibc.loc[ft_ibc['label'] == "Liberal"].head(1250)
neutral = ft_ibc.loc[ft_ibc['label'] == "Neutral"].head(500)

eval_con = ft_ibc.loc[ft_ibc['label'] == "Conservative"].tail(375)
eval_lib = ft_ibc.loc[ft_ibc['label'] == "Liberal"].head(375)

evaluation = shuffle(pd.concat([eval_con, eval_lib]), random_state=3)

train_data = shuffle(pd.concat([conservative, liberal, neutral]), random_state=2)

options = ["liberal", "neutral", "conservative"]

def add_to_dataset(dataset, sentence, label):
    if label == 'liberal':
        result = 0
    elif label == 'neutral':
        result = 1
    else:
        result = 2

    data = {"sentence": sentence,
            "label": result}
    dataset.append(data)

sample_dataset = []

for index in range(len(sample)):
    sentence = sample.iloc[index]["sentence"]
    add_to_dataset(sample_dataset, sentence, sample.iloc[index]["label"].lower())

dataset = []

for index in range(len(train_data)):
    sentence = train_data.iloc[index]["sentence"]
    add_to_dataset(dataset, sentence, train_data.iloc[index]["label"].lower())

evaluation_dataset = []

for index in range(len(evaluation)):
    sentence = evaluation.iloc[index]["sentence"]
    add_to_dataset(evaluation_dataset, sentence, evaluation.iloc[index]["label"].lower())


test_split = (int) (0.2*len(dataset))
test_set = dataset[:test_split]
train_set = dataset[test_split:]
