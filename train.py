import os
import sys
import json
import yaml 
import argparse
from collections import defaultdict

import torch
import wandb

import numpy as np
from tqdm import tqdm

from utils.data_gen import gen_or_load_dataset
from utils.data_utils import load_features, save_json, load_json
from utils.data_loader import get_train_loader, get_test_loader

from modeling.model import build_model
from modeling.loss import build_loss

import utils.evaluate_ego4d_nlq as ego4d_eval



def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config-file", help="Config File with all the parameters", default='config.yaml'
    )
    parser.add_argument('--span_type', default='cw', type=str,
                    choices=['cw', 'xx'],
                    help="Type of span (cw: center-width / xx: start-end)")

    try:
        parsed_args = parser.parse_args()
    except (IOError) as msg:
        parser.error(str(msg))

    # Read config yamls file
    config_file = parsed_args.config_file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if key not in parsed_args.__dict__ or parsed_args.__dict__[key] is None:
            value = value if value != "None" else None
            parsed_args.__dict__[key] = value

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args

def convert_length_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(
        lengths.size()[0], max_len
    ) < lengths.unsqueeze(1)
    mask = mask.float()
    return mask


def main(train_loader, val_loader, config):

    home_dir = os.path.join(
        config.save_model_dir,
        f"{config.model_name}_{config.model_version}_{config.num_querys}_{config.video_strid_length}",
    )
    if config.suffix is not None:
        home_dir = home_dir + f"_{config.suffix}"
    model_dir = os.path.join(home_dir, "model")

    #train
    if config.mode.lower() == "train":

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_json(
            vars(config),
            os.path.join(model_dir, "configs.json"),
            sort_keys=True,
            save_pretty=True,
        )

        model = build_model(config)
        criterion = build_loss(config)
        model.to(config.device)
        criterion.to(config.device)

        head_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                head_params.append(param)
        param_dicts = [{'params':head_params}]

        optimizer = torch.optim.AdamW(param_dicts, lr=config.lr, weight_decay=config.wd)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_step)

        checkpoint_path = os.path.join(model_dir, "checkpoint")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        
        log_path = os.path.join(model_dir, "log")
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        att_vis_path = os.path.join(model_dir, "att_vis")

        print("start training....")
        for epoch in range(config.epochs):
            total_tloss = 0
            total_vloss = 0

            model.train()
            criterion.train()
            for idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train: epoch {epoch+1} / {config.epochs}"):
                records, vfeats, vfeat_lens, qfeats, qfeat_lens, query_flens, targets = data
                v_mask = convert_length_mask(vfeat_lens)
                q_mask = convert_length_mask(qfeat_lens)
                vfeats = vfeats.to(config.device)
                qfeats = qfeats.to(config.device)
                v_mask = v_mask.to(config.device)
                q_mask = q_mask.to(config.device)
                temp_target = []
                temp_target_bool = []
                for t in targets["target_spans"]:
                    temp_target.append(dict(spans=t['spans'].to(config.device)))
                for t in targets["target_bools"]:
                    temp_target_bool.append(dict(spans=t['spans'].to(config.device)))

                targets["target_spans"] = temp_target
                targets["target_bools"] = temp_target_bool

                outputs = model(vfeats, qfeats, v_mask, q_mask, epoch, idx, att_vis_path, "Train")

                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                # if idx % 100 == 0:
                #     print(f"Loss dict: {loss_dict}")

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                loss_dict['loss_overall'] = float(losses)
                total_tloss += loss_dict['loss_overall']

            print(f"Train: Avg loss {total_tloss / len(train_loader)}")  
            print(f"Train: Total loss {total_tloss}")

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }
            torch.save(
                checkpoint,
                os.path.join(checkpoint_path, f"checkpoint_{epoch}.pt")
            )

            results = {}
            model.eval()
            criterion.eval()
            for idx, data in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Val: epoch {epoch+1} / {config.epochs}"):
                records, vfeats, vfeat_lens, qfeats, qfeat_lens, query_flens, targets = data
                v_mask = convert_length_mask(vfeat_lens)
                q_mask = convert_length_mask(qfeat_lens)
                vfeats = vfeats.to(config.device)
                qfeats = qfeats.to(config.device)
                v_mask = v_mask.to(config.device)
                q_mask = q_mask.to(config.device)
                temp_target = []
                for t in targets["target_spans"]:
                    temp_target.append(dict(spans=t['spans'].to(config.device)))
                
                targets["target_spans"] = temp_target

                outputs = model(vfeats, qfeats, v_mask, q_mask, epoch, idx, att_vis_path, "Val")

                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                # if idx % 45 == 0:
                #     print(f"Loss dict: {loss_dict}")

                loss_dict['loss_overall'] = float(losses)
                total_vloss += loss_dict['loss_overall']

                _result = getResults(outputs, records, config)
                for k, v in _result.items():
                    if k not in results:
                        results[k] = list()
                    results[k].append(v)
            
            print(f"Val: Avg loss {total_vloss / len(train_loader)}")  
            print(f"Val: Total loss {total_vloss}")

            with open(os.path.join(log_path, f"val_{epoch}.pt"), 'w') as f:
                json.dump(results, f)

            results, mIoU, score_str = getMetrics(results, epoch)
            print(score_str, flush=True)
            
def getMetrics(results, epoch):
    thresholds = [0.3, 0.5, 0.01]
    topK = [1, 3, 5]
    results, mIoU = ego4d_eval.evaluate_nlq_performance(
        results, thresholds, topK
    )
    title = f"Epoch {epoch}"
    score_str = ego4d_eval.display_results(
        results, mIoU, thresholds, topK, title=title
    )

    return results, mIoU, score_str

def getResults(outputs, records, config):
    import torch.nn as nn
    softmax = nn.Softmax(dim=-1)
    timespans = outputs['pred_spans']  # (batch_size, #queries, 2)
    label_prob = softmax(outputs['pred_logits'])  # (batch_size, #queries, #classes)
    scores, labels = label_prob.max(-1)  # (batch_size, #queries)

    # compose predictions
    results = {}
    
    for span, score, label, record in zip(timespans.cpu(), scores.cpu(), labels.cpu(), records):
        spans = (span * record['number_frame'])
        sorted_preds = torch.cat([label[:, None], spans, score[:, None]], dim=1).tolist()
        sorted_preds = sorted(sorted_preds, key=lambda x: x[3], reverse=True)
        
        sorted_preds = torch.tensor(sorted_preds)
        sorted_labels = sorted_preds[:, 0].int().tolist()
        sorted_spans = sorted_preds[:, 1:].tolist()
        sorted_spans = [[float(f'{e:.4f}') for e in row] for row in sorted_spans]

        for idx, (query_id, queryB) in enumerate(zip(record["query_feature_name"], record["query_bool"])):
            if not queryB:
                continue
            pred_spans = [[ max( 0, (pred_span[0]-pred_span[1])) , min( record['number_frame'], (pred_span[0]+pred_span[1])), pred_span[2]] for pred_label, pred_span in zip(sorted_labels, sorted_spans) if pred_label == idx]
            # _pred_spans = [[ (pred_span[0]-pred_span[1]), (pred_span[0]+pred_span[1]), pred_span[2]] for pred_label, pred_span in zip(sorted_labels, sorted_spans) if pred_label == idx]
            _t_c = record['s_time'][idx]
            _t_w = record['e_time'][idx]
            cur_query_pred = dict(
                            video_id=record['vid'],
                            pred_timespan=pred_spans[:6],
                            gt_span=[record['o_s_time'][idx], record['o_e_time'][idx]],
                            gt_timespan=[ _t_c - _t_w, _t_c + _t_w],
                            v_s_e=record['v_s_e'],   
                        )
            results[query_id] = cur_query_pred

    return results

                
# def getResults(outputs, records, config):
#     import torch.nn as nn
#     softmax = nn.Softmax(dim=-1)
#     timespans = outputs['pred_spans']  # (batch_size, #queries, 2)
#     label_prob = softmax(outputs['pred_logits'])  # (batch_size, #queries, #classes)
#     scores, labels = label_prob.max(-1)  # (batch_size, #queries)

#     # compose predictions
#     results = {}
    
#     for span, score, label, record in zip(timespans.cpu(), scores.cpu(), labels.cpu(), records):
#         spans = ((span * record['number_frame']) + record['v_s_e'][0]) / 1.875
#         sorted_preds = torch.cat([label[:, None], spans, score[:, None]], dim=1).tolist()
#         sorted_preds = sorted(sorted_preds, key=lambda x: x[3], reverse=True)
        
#         sorted_preds = torch.tensor(sorted_preds)
#         sorted_labels = sorted_preds[:, 0].int().tolist()
#         sorted_spans = sorted_preds[:, 1:].tolist()
#         sorted_spans = [[float(f'{e:.4f}') for e in row] for row in sorted_spans]

#         for idx, (query_id, queryB) in enumerate(zip(record["query_feature_name"], record["query_bool"])):
#             if not queryB:
#                 continue
#             pred_spans = [[ max(record['v_s_e'][0] / 1.875, (pred_span[0]-pred_span[1])) , min( (record['v_s_e'][1] / 1.875), (pred_span[0]+pred_span[1])), pred_span[2]] for pred_label, pred_span in zip(sorted_labels, sorted_spans) if pred_label == idx]
#             # _pred_spans = [[ (pred_span[0]-pred_span[1]), (pred_span[0]+pred_span[1]), pred_span[2]] for pred_label, pred_span in zip(sorted_labels, sorted_spans) if pred_label == idx]
#             cur_query_pred = dict(
#                             video_id=record['vid'],
#                             pred_timespan=pred_spans[:6],
#                             gt_span=[record['o_s_time'][idx], record['o_e_time'][idx]],
#                             gt_timespan=[record['exact_s_time'][idx], record['exact_e_time'][idx]],
#                             v_s_e=record['v_s_e'],   
#                         )
#             results[query_id] = cur_query_pred

#     return results

if __name__ == "__main__":
    args = parse_arguments()
    dataset = gen_or_load_dataset(args)

    video_features, query_features, args.vid_feat_dim, args.txt_feat_dim = load_features(
        dataset, os.path.join(args.data_dir, args.video_f_path), os.path.join(args.data_dir, args.query_fmodel),
        args.video_seq_len)

    train_loader = get_train_loader(dataset['train_set'], video_features, query_features, args)
    val_loader = get_test_loader(dataset['val_set'], video_features, query_features, args)
    test_loader = get_test_loader(dataset['test_set'], video_features, query_features, args)

    main(train_loader, val_loader, args)