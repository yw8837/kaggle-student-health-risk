import pandas as pd
import numpy as np

files = [
    'subs/lucifer19/submission.csv',
    'subs/shadowcat/submission.csv',
    'subs/hikari/submission.csv',
    'consensus_src/artkomissar_s6e7-external-consensus-mf-lb-0-95114/submission.csv',
    'consensus_src/hikari30_s6e7-external-ensemble-lb-chase-lb0-95113/submission.csv'
]

ids = pd.read_csv(files[0])['id'].values
classes = ['at-risk', 'unhealthy', 'fit']
cls_to_int = {c: i for i, c in enumerate(classes)}

votes = np.zeros((len(ids), 3), dtype=int)

for f in files:
    df = pd.read_csv(f)
    int_labels = df['health_condition'].map(cls_to_int).values
    votes[np.arange(len(ids)), int_labels] += 1

best_idx = np.argmax(votes, axis=1)
best_labels = [classes[i] for i in best_idx]

sub = pd.DataFrame({'id': ids, 'health_condition': best_labels})
sub.to_csv('subs/ULTIMATE_submission.csv', index=False)
print('Saved ULTIMATE_submission.csv')
