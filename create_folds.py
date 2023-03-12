#%%writefile create_folds.py

from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import GroupKFold


data_dir = Path("./")


# Load CSVs
print("Loading DataFrames")
content_df = pd.read_csv(
    data_dir / "content.csv"
)
corr_df = pd.read_csv(
    data_dir / "correlations.csv"
)
topic_df = pd.read_csv(
    data_dir / "processed_topics.csv"
)

topic_df = topic_df.rename(
    columns={
        "id": "topic_id",
        "title": "topic_title",
        "description": "topic_description",
        "language": "topic_language",
        "context": "topic_context"
    }
)
content_df = content_df.rename(
    columns={
        "id": "content_id",
        "title": "content_title",
        "description": "content_description",
        "text": "content_text",
        "language": "content_language"
    }
)

import re

def clean_text(text):
    # Remove HTML and web links
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+', '', text)
    
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    
    # Clean whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Fill in blanks and limit the amount of content text
topic_df["topic_title"].fillna("", inplace=True)
topic_df["topic_title"] = topic_df["topic_title"].apply(clean_text)
# topic_df["topic_title_len"] = topic_df["topic_title"].apply(lambda x: len(x.split()))
# print('title topic', topic_df["topic_title_len"].describe())

topic_df["topic_description"].fillna("", inplace=True)
topic_df["topic_description"] = topic_df["topic_description"].apply(clean_text)
# topic_df["topic_description_len"] = topic_df["topic_description"].apply(lambda x: len(x.split()))
# print('des topic',topic_df["topic_description_len"].describe())

topic_df["topic_context"].fillna("", inplace=True)
topic_df["topic_context"] = topic_df["topic_context"].apply(clean_text)
# topic_df["topic_context_len"] = topic_df["topic_context"].apply(lambda x: len(x.split()))
# print('context topic', topic_df["topic_context_len"].describe())


####Text final 
topic_df["topic_final_text"] = topic_df["topic_title"] + "[SEP]" + topic_df["topic_description"] + "[SEP]" + topic_df["topic_context"]
topic_df["topic_final_text_len"] = topic_df["topic_final_text"].apply(lambda x: len(x.split()))
print('final topic', topic_df["topic_final_text_len"].describe())

content_df["content_title"].fillna("", inplace=True)
content_df["content_title"] = content_df["content_title"].apply(clean_text)
# content_df["content_title_len"] = content_df["content_title"].apply(lambda x: len(x.split()))
# print('title content', content_df["content_title_len"].describe())

content_df["content_description"].fillna("", inplace=True)
content_df["content_description"] = content_df["content_description"].apply(clean_text)
# content_df["content_description_len"] = content_df["content_description"].apply(lambda x: len(x.split()))
# print('des content', content_df["content_description_len"].describe())

content_df["content_text"].fillna("", inplace=True)
content_df["content_description"] = content_df["content_description"].apply(clean_text)
# content_df["content_text_len"] = content_df["content_text"].apply(lambda x: len(x.split()))
# print('text content', content_df["content_text_len"].describe())
content_df["content_text"] = [x[:300] for x in content_df["content_text"]]

content_df["content_final_text"] = content_df["content_title"] + "[SEP]" + content_df["content_description"] + "[SEP]" + content_df["content_text"]
content_df["content_final_text_len"] = content_df["content_final_text"].apply(lambda x: len(x.split()))
print('final content', content_df["content_final_text_len"].describe())

print()
# `exploded` has one topic id and one content id per row
corr_df["content_id"] = [x.split() for x in corr_df["content_ids"]]
exploded = corr_df.explode("content_id")


# I'll use theoviel's approach for creating a CV
# https://www.kaggle.com/code/theoviel/modeling-oriented-eda-building-a-good-cv-split

# Ignore source because they won't be in test set
topics_val = topic_df[topic_df["category"] != "source"][["channel", "topic_id"]]
topics_val = topics_val.merge(corr_df, on="topic_id")

channel_val = topics_val.groupby("channel").agg(list).reset_index()
channel_val["content_ids"] = channel_val["content_ids"].apply(
    lambda x: list(np.unique(np.concatenate([x_.split(" ") for x_ in x])))
)


def iou(a, b):
    return len(set(a).intersection(set(b))) / len(set(a + b))


print("Getting ious")
ious = np.zeros((len(channel_val), len(channel_val)))

# Get measure of overlap in content ids between channels
for i in range(len(channel_val)):
    for j in range(i):
        iou_ij = iou(channel_val["content_ids"][i], channel_val["content_ids"][j])
        ious[i, j] = iou_ij
        ious[j, i] = iou_ij


G = nx.Graph(ious)  # you can also threshold ious
components = list([list(k) for k in nx.connected_components(G)])

# `components` has one element for each isolated graph
# one element may have multiple channels but
# there are no content ids in common across components
for i, c in enumerate(components):
    # Number of channels in each component
    print(f"Component {i}: {len(c)}")


# Assign a group to each element of `component`
channel_val["group"] = 0
for i, c in enumerate(components):
    channel_val.loc[np.array(c), "group"] = i


# Merge with other dataframes so each content_id has an associated group

combined = topic_df.merge(exploded, on="topic_id").merge(content_df, on="content_id")
combined = combined.merge(channel_val[["channel", "group"]], on="channel", how="left")
combined["fold"] = -1

print("Combined shape:", combined.shape)

# rows with category==source will be nan in the group column
groups = combined.loc[~combined.group.isna()].reset_index(drop=True)

# Group KFold ensures no channels or content ids are in multiple folds
gkf = GroupKFold(n_splits=4)
for fold, (_, val_idx) in enumerate(
    gkf.split(groups, groups=groups["group"].astype(int))
):
    groups.loc[val_idx, "fold"] = fold

combined.loc[~combined.group.isna(), "fold"] = groups.fold.values

print("Fold counts")
print(combined["fold"].value_counts())

combined.to_parquet("combined1.pq", index=False)
