import os

with open('src/A_3.py', 'r', encoding='utf-8') as f:
    code = f.read()

import_env = """import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

"""

if 'OMP_NUM_THREADS' not in code:
    code = import_env + code

with open('src/A_3.py', 'w', encoding='utf-8') as f:
    f.write(code)
