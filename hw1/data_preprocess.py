import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
cancer_reg = pd.read_csv("./cancer_reg.csv", encoding="ISO-8859-1")
cancer_reg.info()

# Dedup data set
cancer_reg.drop_duplicates(inplace=True)

# Split 'binnedInc' column
if 'binnedInc' in cancer_reg:
  split_binned_inc = cancer_reg['binnedInc'].str.extract(r"^[\(\[](.*), (.*)[\)\]]$", expand=True)
  split_binned_inc.rename(columns={0: 'binnedIncLow', 1: 'binnedIncHigh'}, inplace=True)
  split_binned_inc['binnedIncLow'] = pd.to_numeric(split_binned_inc['binnedIncLow'], errors='coerce')
  split_binned_inc['binnedIncHigh'] = pd.to_numeric(split_binned_inc['binnedIncHigh'], errors='coerce')
  cancer_reg = pd.concat([cancer_reg, split_binned_inc], axis=1)
  cancer_reg.drop(columns=['binnedInc'], inplace=True)

# Delete 'geography' column
if 'Geography' in cancer_reg:
  cancer_reg.drop(columns=['Geography'], inplace=True)
# Opt: Do word embedding of 'geography' column

# Delete features with missing data
cancer_reg.dropna(axis=1, inplace=True)

# min-max normalize data
cancer_reg = (cancer_reg - cancer_reg.min())/(cancer_reg.max() - cancer_reg.min())


# Split data into training, validation, and test sets
train, test = train_test_split(cancer_reg, test_size=0.1)
train, validation = train_test_split(train, test_size=0.1111111)

# Save training, validation, and test sets each into their own .csv file
train.to_csv('train.csv', index=False)
validation.to_csv('validation.csv', index=False)
test.to_csv('test.csv', index=False)