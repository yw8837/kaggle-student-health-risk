import pandas as pd
import numpy as np

files = [
    'subs/lucifer19/submission.csv',
    'subs/shadowcat/submission.csv',
    'subs/hikari/submission.csv',
    'consensus_src/artkomissar_s6e7-external-consensus-mf-lb-0-95114/submission.csv',
    'consensus_src/hikari30_s6e7-external-ensemble-lb-chase-lb0-95113/submission.csv'
]

dfs = [pd.read_csv(f).set_index('id') for f in files]
merged = pd.concat(dfs, axis=1)

# Hard voting
mode_result = merged.mode(axis=1)[0]
mode_result.name = 'health_condition'

mode_result.reset_index().to_csv('subs/ULTIMATE_submission.csv', index=False)
print('Saved ULTIMATE_submission.csv')
