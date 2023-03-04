#%%writefile fix_content_ids.py

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