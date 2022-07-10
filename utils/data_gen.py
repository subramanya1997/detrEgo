import os
import sys
import multiprocessing
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from utils.data_utils import (
    load_pickle, save_pickle, load_json, save_json
    )

class EpisodicNLQProcessor:
    def __init__(self, max_pos_len, num_querys, video_seq_len):
        super(EpisodicNLQProcessor, self).__init__()
        self.idx_counter = 0
        self.max_pos_len = max_pos_len
        self.num_querys = num_querys
        self.video_seq_len = video_seq_len

    def reset_idx_counter(self):
        self.idx_counter = 0

    def process_data_tan(self, data, scope, vfeat_lens):
        results = []
        not_vid = 0

        for vid, data_item in tqdm(
            data.items(), total=len(data), desc=f"process episodic nlq {scope}"
            ):
            if vid not in vfeat_lens:
                raise ValueError(f"{vid} not in vfeat_lens")
            if vfeat_lens[vid] > self.max_pos_len:
                not_vid += 1
                continue

            num_frames = vfeat_lens[vid] // self.video_seq_len if vfeat_lens[vid] % self.video_seq_len == 0 else (vfeat_lens[vid] // self.video_seq_len) + 1

            zipper = list(zip(
                data_item["timestamps"],
                data_item["exact_times"],
                data_item["sentences"],
                data_item["annotation_uids"],
                data_item["query_idx"],
            ))

            #need to work on this
            if len(zipper) < self.num_querys:
                # print(f"{vid} has less than {self.num_querys} querys")
                not_vid += 1
                continue

            for i in range(0, len(zipper), self.num_querys):
                
                record = {
                    "sample_id": self.idx_counter,
                    "vid": str(vid),
                    "s_time": [],
                    "e_time": [],
                    "exact_s_time": [],
                    "exact_e_time": [],
                    "query": [],
                    "query_feature_name": [],
                    "annotation_uid": [],
                    "query_idx": [],
                    "num_frames": num_frames,
                    "video_frames_len": vfeat_lens[vid],
                    "annotation_length": [],
                }
                temp_data = zipper[i:i+self.num_querys]
                if len(temp_data) != self.num_querys:
                    _t = i - (self.num_querys - len(temp_data))
                    temp_data = zipper[_t:_t+self.num_querys]

                for _data in temp_data:
                    _s = _data[0][0] // self.video_seq_len
                    _e = _data[0][1] // self.video_seq_len
                    _w = (_e - _s) / 2 
                    _c = (_s + _e) / 2
                    record["s_time"].append(_c)
                    record["e_time"].append(_w)
                    record["exact_s_time"].append(_data[1][0])
                    record["exact_e_time"].append(_data[1][1])
                    record["query"].append(_data[2].strip().lower())
                    _annotation_length = (_data[0][1] - _data[0][0]) + 1
                    record["query_feature_name"].append(f"{_data[3]}_{_data[4]}")
                    record["annotation_uid"].append(_data[3])
                    record["query_idx"].append(_data[4])
                    record["annotation_length"].append(_annotation_length)
                results.append(record)
                self.idx_counter += 1
        print(f"{not_vid} videos are too long")
        print(f"{self.idx_counter} samples in total")
        return results

    def convert(self, data_dir, vfeat_lens):
        if not os.path.exists(data_dir):
            raise ValueError("data dir {} does not exist".format(data_dir))

         # load raw data
        train_data = load_json(os.path.join(data_dir, "train.json"))
        val_data = load_json(os.path.join(data_dir, "val.json"))
        test_data = load_json(os.path.join(data_dir, "test.json"))

        # process data
        self.reset_idx_counter()
        train_set = self.process_data_tan(train_data, "train", vfeat_lens)
        self.reset_idx_counter()
        val_set = self.process_data_tan(val_data, "val", vfeat_lens)
        self.reset_idx_counter()
        test_set = self.process_data_tan(test_data, "test", vfeat_lens)
        self.reset_idx_counter()
        return train_set, val_set, test_set

class EpisodicNLQSplitProcessor:
    def __init__(self, max_pos_len, num_querys, video_seq_len, video_split_length, video_strid_length):
        super(EpisodicNLQSplitProcessor, self).__init__()
        self.idx_counter = 0
        self.max_pos_len = max_pos_len
        self.num_querys = num_querys
        self.video_seq_len = video_seq_len
        self.video_split_length = video_split_length
        self.video_strid_length = video_strid_length

    def reset_idx_counter(self):
        self.idx_counter = 0

    def process_data_tan(self, data, scope, vfeat_lens, video_split_length, video_strid_length):
        results = []
        not_vid = 0

        for vid, data_item in tqdm(
            data.items(), total=len(data), desc=f"process episodic nlq {scope}"
            ):
            if vid not in vfeat_lens:
                raise ValueError(f"{vid} not in vfeat_lens")
            if vfeat_lens[vid] > self.max_pos_len:
                not_vid += 1
                continue

            num_frames = vfeat_lens[vid] // self.video_seq_len if vfeat_lens[vid] % self.video_seq_len == 0 else (vfeat_lens[vid] // self.video_seq_len) + 1

            zipper = list(zip(
                data_item["timestamps"],
                data_item["exact_times"],
                data_item["sentences"],
                data_item["annotation_uids"],
                data_item["query_idx"],
            ))
            zipper = sorted(zipper, key=lambda x: x[0][1])

            #need to work on this
            if len(zipper) < self.num_querys:
                # print(f"{vid} has less than {self.num_querys} querys")
                not_vid += 1
                continue

            for i in range(0, len(zipper)-4, 1):
                for _i, s_f in enumerate(range(0, num_frames, video_strid_length)):
                    e_f = s_f + video_split_length if ((s_f + video_split_length) <= num_frames) else num_frames
                    record = {
                        "sample_id": self.idx_counter,
                        "vid": str(vid),
                        "s_time": [],
                        "e_time": [],
                        "o_s_time": [],
                        "o_e_time": [],
                        "exact_s_time": [],
                        "exact_e_time": [],
                        "query": [],
                        "query_feature_name": [],
                        "query_bool": [],
                        "annotation_uid": [],
                        "query_idx": [],
                        "number_frame": video_split_length,
                        "num_frames": num_frames,
                        "video_frames_len": vfeat_lens[vid],
                        "annotation_length": [],
                        "v_s_e": [s_f, e_f],
                        "vid_part": _i,
                    }
                    temp_data = zipper[i:i+self.num_querys]
                    if len(temp_data) != self.num_querys:
                        _t = i - (self.num_querys - len(temp_data))
                        temp_data = zipper[_t:_t+self.num_querys]

                    _cnt = 0
                    for _data in temp_data:
                        _s = _data[0][0] 
                        _e = _data[0][1]
                        _start = 0
                        _end = 0
                        if abs(_s - s_f) <= video_split_length:
                            _start = max(_start, _s - s_f)
                            _end = min( (video_split_length - 1), _e - s_f)
                            if _end < 0:
                                _end = 0
                        _tempb = True
                        if _start == 0 and _end == 0:
                            _cnt += 1
                            _tempb = False

                        _w = (_end - _start) / 2 
                        _c = (_end + _start) / 2
                        record["s_time"].append(_c)
                        record["e_time"].append(_w)
                        record["o_s_time"].append(_s)
                        record["o_e_time"].append(_e)
                        record["exact_s_time"].append(_data[1][0])
                        record["exact_e_time"].append(_data[1][1])
                        record["query"].append(_data[2].strip().lower())
                        _annotation_length = (_data[0][1] - _data[0][0]) + 1
                        record["query_feature_name"].append(f"{_data[3]}_{_data[4]}")
                        record["query_bool"].append(_tempb)
                        record["annotation_uid"].append(_data[3])
                        record["query_idx"].append(_data[4])
                        record["annotation_length"].append(_annotation_length)
                    if _cnt == len(temp_data) and scope != "val":
                        continue
                    results.append(record)
                    self.idx_counter += 1

        print(f"{not_vid} videos are too long")
        print(f"{self.idx_counter} samples in total")
        return results

    def convert(self, data_dir, vfeat_lens):
        if not os.path.exists(data_dir):
            raise ValueError("data dir {} does not exist".format(data_dir))

         # load raw data
        train_data = load_json(os.path.join(data_dir, "train.json"))
        val_data = load_json(os.path.join(data_dir, "val.json"))
        test_data = load_json(os.path.join(data_dir, "test.json"))

        # process data
        self.reset_idx_counter()
        train_set = self.process_data_tan(train_data, "train", vfeat_lens, self.video_split_length, self.video_strid_length)
        self.reset_idx_counter()
        val_set = self.process_data_tan(val_data, "val", vfeat_lens, self.video_split_length, self.video_split_length)
        self.reset_idx_counter()
        test_set = self.process_data_tan(test_data, "test", vfeat_lens, self.video_split_length, self.video_split_length)
        self.reset_idx_counter()
        return train_set, val_set, test_set

def save_query_features(model_name, data_dir, data, scope, device="cpu"):
    from transformers import AutoTokenizer, AutoModel
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    _save_dir = os.path.join(data_dir, model_name)
    if not os.path.exists(_save_dir):
        os.makedirs(_save_dir)

    description = f"process {scope} query features"
    for record in tqdm(data, total=len(data), desc=description):    
        query_zipper = list(zip(
            record["query"],
            record["query_feature_name"]
            ))
        for query, query_name in query_zipper:
            _path = os.path.join(_save_dir, query_name)
            if os.path.exists(f"{_path}.pt"):
                continue
            inputs = tokenizer(query, return_tensors="pt").to(device)
            outputs = model(**inputs).last_hidden_state
            torch.save(outputs.cpu(), f"{_path}.pt")
    
def gen_or_load_dataset(config):
    
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    data_dir = config.data_dir
    if not os.path.exists(data_dir):
        raise ValueError("data dir {} does not exist".format(data_dir))

    video_f_dir = os.path.join(data_dir, config.video_f_path)

    if config.suffix is None:
        save_path = os.path.join(config.save_dir, f"dataset_{config.num_querys}_{config.query_fmodel}_{config.video_seq_len}")
    else:
        save_path = os.path.join(config.save_dir, f"dataset_{config.num_querys}_{config.query_fmodel}_{config.video_seq_len}_{config.suffix}")
    
    if config.video_split:
        save_path = f"{save_path}_{config.video_split_length}_{config.model_version}_{config.video_strid_length}"
    save_path = f"{save_path}.pt"

    print(f"save_path: {save_path}")
    if os.path.exists(save_path):
        return load_pickle(save_path)
    
    feat_len_path = os.path.join(video_f_dir, "feature_shapes.json")
    vfeat_lens = load_json(feat_len_path)

    # load data
    if not config.video_split:
        processor = EpisodicNLQProcessor(config.max_pos_len, config.num_querys, config.video_seq_len)
    else:
        processor = EpisodicNLQSplitProcessor(config.max_pos_len, config.num_querys, config.video_seq_len, config.video_split_length, config.video_strid_length)

    train_data, val_data, test_data = processor.convert(
        data_dir, vfeat_lens
    )

    save_query_features(config.query_fmodel, data_dir, train_data, "train", config.device)
    save_query_features(config.query_fmodel, data_dir, val_data, "val", config.device)
    save_query_features(config.query_fmodel, data_dir, test_data, "test", config.device)

    dataset = {
        "train_set": train_data,
        "val_set": val_data,
        "test_set": test_data,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "vfeat_len": None,
        "qfeat_len": None
    }

    save_pickle(dataset, save_path)
    return dataset