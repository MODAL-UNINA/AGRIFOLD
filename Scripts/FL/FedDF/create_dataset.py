# -*- coding: utf-8 -*-
import torch

from pcode.datasets.partition_data import DataPartitioner
from pcode.datasets.prepare_data import get_dataset
import pcode.datasets.mixup_data as mixup


"""create dataset and load the data_batch."""


def load_data_batch(conf, _input, _target, is_training=True):
    """Load a mini-batch and record the loading time."""
    if conf.graph.on_cuda:
        _input, _target = _input.cuda(), _target.cuda()

    # argument data.
    if conf.use_mixup and is_training:
        _input, _target_a, _target_b, mixup_lambda = mixup.mixup_data(
            _input,
            _target,
            alpha=conf.mixup_alpha,
            assist_non_iid=conf.mixup_noniid,
            use_cuda=conf.graph.on_cuda,
        )
        _data_batch = {
            "input": _input,
            "target_a": _target_a,
            "target_b": _target_b,
            "mixup_lambda": mixup_lambda,
        }
    else:
        _data_batch = {"input": _input, "target": _target}
    return _data_batch


# %%
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, file_paths, class_labels, transform=None):

        self.file_paths = file_paths
        self.class_labels = class_labels
        self.transform = transform

        self.label_map = {label: idx for idx, label in enumerate(self.class_labels)}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        import numpy as np
        file_path = self.file_paths[idx]

        label = None
        for class_name in self.class_labels:
            if f"/{class_name}/" in file_path:
                label = self.label_map[class_name]
                break
        if label is None:
            raise ValueError(f"Etichetta non trovata per il file: {file_path}")

        data = np.load(file_path)
        image = data[
            "normalized_image"
        ]

        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)

        return image, label
    
def load_data(dataset, batch_size):
    from torch.utils.data import DataLoader
    

    split_data = balance_data_from_json(dataset, val_ratio=0, random_seed=42)

    print(f"Train set: {len(split_data['train'])} immagini")
    print(f"Validation set: {len(split_data['val'])} immagini")
    print(f"Test set: {len(split_data['test'])} immagini")

    class_labels = list(dataset.keys())

    train_dataset = CustomDataset(split_data["train"], class_labels)
    val_dataset = CustomDataset(split_data["val"], class_labels)
    test_dataset = CustomDataset(split_data["test"], class_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def balance_data_from_json(
    json_data, train_ratio=0.7, val_ratio=0.1, random_seed=42
    ):
    from sklearn.model_selection import train_test_split

    splits = {"train": [], "val": [], "test": []}

    for disease, v in json_data.items():
        for fruit, paths in v.items():

            if len(paths) == 0:
                continue 
            train_paths, temp_paths = train_test_split(
                paths, train_size=train_ratio, random_state=random_seed
            )
            test_paths = temp_paths

            splits["train"].extend(train_paths)
            splits["test"].extend(test_paths)

        print(
            f"{disease}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test"
        )
    print(
        f"Total: {len(splits['train']) + len(splits['val']) + len(splits['test'])} images"
    )
    return splits

def define_dataset(conf, data, display_log=True):
    # prepare general train/test.
    conf.partitioned_by_user = True if "femnist" == conf.data or "plants" == conf.data else False
    train_dataset = get_dataset(conf, data, conf.data_dir, split="train")
    test_dataset = get_dataset(conf, data, conf.data_dir, split="test")

    if conf.data != 'plants':
        # create the validation from train.
        train_dataset, val_dataset, test_dataset = define_val_dataset(
            conf, train_dataset, test_dataset
        )
    else:
        val_dataset = None
        print(f"No validation dataset for {conf.data} dataset.")

    if display_log:
        conf.logger.log(
            "Data stat for original dataset: we have {} samples for train, {} samples for val, {} samples for test.".format(
                len(train_dataset),
                len(val_dataset) if val_dataset is not None else 0,
                len(test_dataset),
            )
        )
    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}


def define_val_dataset(conf, train_dataset, test_dataset):
    assert conf.val_data_ratio >= 0

    partition_sizes = [
        (1 - conf.val_data_ratio) * conf.train_data_ratio,
        (1 - conf.val_data_ratio) * (1 - conf.train_data_ratio),
        conf.val_data_ratio,
    ]

    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="origin",
        consistent_indices=False,
    )
    train_dataset = data_partitioner.use(0)

    # split for val data.
    if conf.val_data_ratio > 0:
        assert conf.partitioned_by_user is False

        val_dataset = data_partitioner.use(2)
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, None, test_dataset



def define_data_loader(
    conf, dataset, localdata_id=None, is_train=True, shuffle=True, data_partitioner=None
):
    # determine the data to load,
    # either the whole dataset, or a subset specified by partition_type.

    import json
    root = "datasets/PlantDataset/"
    file_path = root + "load.json"
    with open(file_path, "r") as d:
        dic = json.load(d)

    disease_data_path = root + "Preprocessing/Preprocessed_dictionary.json"

    with open(disease_data_path, "r") as f:
        disease_data = json.load(f)
    
    client_id = localdata_id
    print("Client ID:", client_id)
    dataset = dic[str(client_id)]

    print("Selected dataset:", dataset)

    dataset = disease_data[dataset]
    sum_images = 0
    for disease, v in dataset.items():
        for fruit, paths in v.items():
            sum_images += len(paths)
    print(f"Totale: {sum_images} immagini")
    class_labels = list(dataset.keys())

    num_classes = len(class_labels)
    train_loader, _, test_loader = load_data(dataset, batch_size=conf.batch_size)

    if is_train:
        data_loader = train_loader
    else:
        data_loader = test_loader

    conf.num_batches_per_device_per_epoch = len(data_loader)
    conf.num_whole_batches_per_worker = (
        conf.num_batches_per_device_per_epoch * conf.local_n_epochs
    )
    return data_loader, data_partitioner
