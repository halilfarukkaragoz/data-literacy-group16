import torch
import numpy as np
import random, os
from datetime import datetime

# df[col] = pd.to_datetime(df[col], errors='coerce', format="%b %d '%y %I:%M %p %Z")

# df[f'{col}_difference'] = df["deadline"] - df[col]
# df[f'{col}_difference'] = df[f'{col}_difference'].dt.total_seconds() / 3600

# print(df[col])
# print(df["deadline"])
# print(df[f'{col}_difference'])

# # Option 1: Convert to timestamp (seconds since epoch)
# # Handle potential NaT by filling with 0 or another placeholder
# df[f'{col}_difference'] = df[col].astype(np.int64) / 1e9
# df[f'{col}_difference'] = df[f'{col}_difference'].replace([np.inf, -np.inf], np.nan)


def seed_everything(seed: int):
        
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def time_difference(time1: datetime, time2: datetime):
    diff = time1 - time2
    return diff.total_seconds() / 3600
