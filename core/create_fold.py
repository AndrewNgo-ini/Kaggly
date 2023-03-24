%%writefile create_folds.py

from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import GroupKFold


data_dir = Path("/kaggle/input/learning-equality-curriculum-recommendations")


# Load CSVs
print("Loading DataFrames")
content_df = pd.read_csv(
    data_dir / "content.csv"
)
corr_df = pd.read_csv(
    data_dir / "correlations.csv"
)
topic_df = pd.read_csv(
    data_dir / "topics.csv"
)

topic_df = topic_df.rename(
    columns={
        "id": "topic_id",
        "title": "topic_title",
        "description": "topic_description",
        "language": "topic_language",
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

# Fill in blanks and limit the amount of content text
topic_df["topic_title"].fillna("No topic title", inplace=True)
topic_df["topic_description"].fillna("No topic description", inplace=True)
content_df["content_title"].fillna("No content title", inplace=True)
content_df["content_description"].fillna("No content description", inplace=True)
content_df["content_text"].fillna("No content text", inplace=True)
content_df["content_text"] = [x[:300] for x in content_df["content_text"]]

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


# %%writefile fix_content_ids.py

import pandas as pd
import json

combined = pd.read_parquet("combined1.pq")

# This will list all the folds that a content_id is in.
# Ideally, it should only be in one fold
g = combined.groupby("content_id")["fold"].agg(list)

print("Before fixing folds")
print(g.head(5), "\n\n")

# This should only have 1 value because each row
# should only have 1 unique value in the list
print("Checking number of folds for each content_id.", set([len(set(x)) for x in g.values]), "\n\n")


# Change the fold of every row with a content_id overlapping with the 
# content ids in category==source to -1
source_c_ids = set(combined[combined["category"]=="source"]["content_id"])
mask = combined["content_id"].isin(source_c_ids)
combined.loc[mask, "fold"] = -1

# Let's check again
g = combined.groupby("content_id")["fold"].agg(list)

print("After fixing folds")
print(g.head(5), "\n\n")

# This should only have 1 value because each row
# should only have 1 unique value in the list
print("Checking number of folds for each content_id.", set([len(set(x)) for x in g.values]), "\n\n")

# This will be used to index the whole dataset later
fold_idxs = {int(f): combined[combined["fold"]==f].index.tolist() for f in combined['fold'].unique()}

# Save it to a file for later so this doesn't have to be run again
with open("training_idxs.json", "w") as fp:
    json.dump(fold_idxs, fp)

combined.to_parquet("combined2.pq", index=False)

print("Fold value counts")
print(combined.fold.value_counts())