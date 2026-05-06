from datasets import load_dataset
import pandas as pd
from sklearn.utils import resample

# Load original HuggingFace dataset
hf_dataset = load_dataset("community-datasets/tamilmixsentiment")
df_hf_train = pd.DataFrame(hf_dataset['train'])
df_hf_test = pd.DataFrame(hf_dataset['test'])
df_hf_val = pd.DataFrame(hf_dataset['validation'])

# Map HuggingFace numeric labels to text
hf_label_map = {0: 'Positive', 1: 'Negative', 2: 'Mixed_feelings', 3: 'unknown_state', 4: 'not-Tamil'}
df_hf_train['label'] = df_hf_train['label'].map(hf_label_map)
df_hf_test['label'] = df_hf_test['label'].map(hf_label_map)
df_hf_val['label'] = df_hf_val['label'].map(hf_label_map)

# Load new Zenodo dataset
df_z_train = pd.read_csv('data/raw/tamil_sentiment_full_train.csv', sep='\t', header=None, names=['text', 'label'], on_bad_lines='skip', engine='python')
df_z_test = pd.read_csv('data/raw/tamil_sentiment_full_test.csv', sep='\t', header=None, names=['text', 'label'], on_bad_lines='skip', engine='python')
df_z_dev = pd.read_csv('data/raw/tamil_sentiment_full_dev.csv', sep='\t', header=None, names=['text', 'label'], on_bad_lines='skip', engine='python')

print("Zenodo label distribution:")
print(df_z_train['label'].value_counts())

# Combine both datasets
df_train_all = pd.concat([df_hf_train[['text','label']], df_z_train], ignore_index=True)
df_test_all = pd.concat([df_hf_test[['text','label']], df_z_test], ignore_index=True)
df_val_all = pd.concat([df_hf_val[['text','label']], df_z_dev], ignore_index=True)

print(f"\nCombined total - Train: {len(df_train_all)} | Test: {len(df_test_all)} | Val: {len(df_val_all)}")
print("\nCombined label distribution:")
print(df_train_all['label'].value_counts())

# Keep only Positive and Negative
df_train_all = df_train_all[df_train_all['label'].isin(['Positive', 'Negative'])].reset_index(drop=True)
df_test_all = df_test_all[df_test_all['label'].isin(['Positive', 'Negative'])].reset_index(drop=True)
df_val_all = df_val_all[df_val_all['label'].isin(['Positive', 'Negative'])].reset_index(drop=True)

# Normalize labels
df_train_all['sentiment'] = df_train_all['label'].str.lower()
df_test_all['sentiment'] = df_test_all['label'].str.lower()
df_val_all['sentiment'] = df_val_all['label'].str.lower()

print(f"\nAfter binary filter - Train: {len(df_train_all)} | Test: {len(df_test_all)} | Val: {len(df_val_all)}")
print(df_train_all['sentiment'].value_counts())

# Oversample negative to match positive
df_positive = df_train_all[df_train_all['sentiment'] == 'positive']
df_negative = df_train_all[df_train_all['sentiment'] == 'negative']

df_negative_upsampled = resample(
    df_negative,
    replace=True,
    n_samples=len(df_positive),
    random_state=42
)

df_train_balanced = pd.concat([df_positive, df_negative_upsampled])
df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal balanced train size: {len(df_train_balanced)}")
print(df_train_balanced['sentiment'].value_counts())

df_train_balanced.to_csv('data/processed/train.csv', index=False)
df_test_all.to_csv('data/processed/test.csv', index=False)
df_val_all.to_csv('data/processed/val.csv', index=False)
print("\nData saved!")