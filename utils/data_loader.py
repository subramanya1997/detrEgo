import numpy as np
import torch
import torch.utils.data

from utils.data_utils import pad_seq

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_features, query_features, configs):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.video_features = video_features
        self.query_features = query_features
        self.configs = configs

    def __getitem__(self, index):
        record = self.dataset[index]
        video_feature = self.video_features[record["vid"]]
        query_feature = [self.query_features[qid] for qid in record["query_feature_name"]]
        
        query_flen = [query.shape[0] for query in query_feature]
        query_feature = torch.cat(query_feature, dim=1)
        target = []
        for s, e in zip(record["s_time"], record["e_time"]):
            target.append([ s / record['num_frames'] , e  / record['num_frames']])
        target = torch.tensor(target)
        return record, video_feature, query_feature[0], query_flen, target

    def __len__(self):
        return len(self.dataset)

def collate_fn(batch):
    records, video_features, query_features, query_flens, targets = zip(*batch)

    vfeats, vfeat_lens = pad_seq(video_features)
    vfeats = torch.stack(vfeats)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)

    qfeats, qfeat_lens = pad_seq(query_features)
    qfeats = torch.stack(qfeats)
    qfeat_lens = torch.tensor(qfeat_lens, dtype=torch.int64)

    targ = {}
    targ['target_spans'] = [ dict(spans=d) for d in targets]
    query_flens = torch.tensor(query_flens)
    return records, vfeats, vfeat_lens, qfeats, qfeat_lens, query_flens, targ

def get_train_loader(dataset, video_features, query_features, configs):
    train_set = Dataset(dataset=dataset, video_features=video_features, query_features=query_features, configs=configs)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=configs.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    return train_loader

def get_test_loader(dataset, video_features, query_features, configs):
    test_set = Dataset(dataset=dataset, video_features=video_features, query_features=query_features, configs=configs)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=configs.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    return test_loader