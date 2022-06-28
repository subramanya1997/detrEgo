## Preparation

### Dataset 

Download the dataset from the [official webpage][ego4d_page] and place them
in `data/` folder.

Run the preprocessing script using:

```bash
python utils/prepare_ego4d_dataset.py \
    --input_train_split data/nlq_train.json \
    --input_val_split data/nlq_val.json \
    --input_test_split data/nlq_test_unannotated.json \
    --video_feature_read_path data/features/nlq_official_v1/video_features/ \
    --clip_feature_save_path data/features/nlq_official_v1/official \
    --output_save_path data/dataset/nlq_official_v1
```

This creates JSON files in `data/dataset/nlq_official_v1` that can be used for training and evaluating the VSLNet baseline model.


### Video features

Download the official video features released from [official webpage][ego4d_page] and place them in `data/features/nlq_official_v1/video_features/` folder.

update config
