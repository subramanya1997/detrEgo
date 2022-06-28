import glob
import json
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

import math

def load_json(filename):
    with open(filename, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, mode="w", encoding="utf-8") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)

def load_pickle(filename):
    with open(filename, mode="rb") as handle:
        data = pickle.load(handle)
        return data


def save_pickle(data, filename):
    with open(filename, mode="wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pad_seq(sequences):
    max_len = max([v.shape[0] for v in sequences])
    feature_length = sequences[0].shape[1]
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        add_length = max_len - seq.shape[0]
        sequence_length.append(seq.shape[0])
        if add_length > 0:
            add_feature = torch.zeros((add_length, feature_length))
            seq_ = torch.cat([seq, add_feature], dim=0)
        else:
            seq_ = seq
        sequence_padded.append(seq_)
    return sequence_padded, sequence_length
    

def load_features(data, vf_path, qf_path, video_seq_len):
    video_features = dict()
    query_features = dict()

    vid_set = set()
    qid_set = set()

    _dataset_names = ["train_set", "val_set", "test_set"]
    for _name in _dataset_names:
        for _data in data[_name]:
            vid_set.add(_data["vid"])
            for _qname in _data["query_feature_name"]:
                qid_set.add(_qname)
    vfeature_len, qfeature_len = None, None
    for _vid in tqdm(vid_set, total=len(vid_set), desc="load video features"):
        vid_fpath = os.path.join(vf_path, f"{_vid}.pt")
        _vfeature = torch.load(vid_fpath)
        _temp = []
        for i in range(0, len(_vfeature), video_seq_len):
            _temp.append(torch.mean(_vfeature[i:i+video_seq_len], dim=0, keepdim=True))
        video_features[_vid] = torch.cat(_temp, dim=0)
        if vfeature_len is None:
            vfeature_len = video_features[_vid].shape[1]

    for _qid in tqdm(qid_set, total=len(qid_set), desc="load query features"):
        qid_fpath = os.path.join(qf_path, f"{_qid}.pt")
        query_features[_qid] = torch.load(qid_fpath)
        if qfeature_len is None:
            qfeature_len = query_features[_qid].shape[2]
    
    return video_features, query_features, vfeature_len, qfeature_len