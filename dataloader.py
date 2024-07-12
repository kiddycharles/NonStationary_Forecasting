import os
import os.path

import pandas as pd
from torch.utils.data import Dataset


def get_global_values(df):
    col = df.columns
    min_value = min(df[col[0]])
    max_value = max(df[col[0]])
    return min_value, max_value


def minmax_scaling(x, min_value, max_value):
    x[:, 0] = (x[:, 0] - min_value) / (max_value - min_value)
    return x


class StockDataset(Dataset):
    def __init__(self, target_dir, file_name, seq_size=(100, 50, 25), args=None, loader_type='train', features='S',
                 transform=None):
        self.seq_len = seq_size[0]    # Sequence length for learning
        self.label_len = seq_size[1]  # Sequence length for guiding
        self.pred_len = seq_size[2]   # Sequence length for predicting
        self.transform = transform    # Normalization
        self.args = args

        file_path = os.path.join(target_dir, file_name)
        data = pd.read_csv(file_path, index_col=0)

        if features == 'S':
            data = data[['close']]

        if file_name == 'AMD.csv' or file_name == 'NVDA.csv':
            data.index = pd.to_datetime(data.index).tz_convert(None)
            train_data = data.truncate(after='2019-09-01 00:00:00')
            test_data = data[len(train_data) - self.seq_len:]
        else:
            data.index = pd.to_datetime(data.index)
            train_data = data.truncate(after='2015-12-31 00:00:00')
            test_data = data[len(train_data) - self.seq_len:]

        min_value, max_value = get_global_values(train_data)

        self.min_value = min_value
        self.max_value = max_value

        if loader_type == 'train':
            data = train_data
        elif loader_type == 'test':
            data = test_data

        self.data = minmax_scaling(data.values, min_value, max_value)

        indices = []
        # stride 10
        for i in range(len(self.data)):
            indices.append(i)

        self.indices = indices[:-(self.seq_len + self.pred_len) + 1]

    def get_minmax(self):
        return self.min_value, self.max_value

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        s_begin = self.indices[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.args.output_attention:
            x = self.data[s_begin:s_end, 0].reshape(-1, 1)
            y = self.data[r_begin: r_end, 0].reshape(-1, 1)
        else:
            x = self.data[s_begin:s_end, 0].reshape(-1, 1)
            y = self.data[s_end: r_end, 0].reshape(-1, 1)

        if self.transform:
            x = self.transform(x)

        return x, y
