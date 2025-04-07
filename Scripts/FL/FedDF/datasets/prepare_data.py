# -*- coding: utf-8 -*-
import os

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


import pcode.datasets.loader.imagenet_folder as imagenet_folder
import pcode.datasets.loader.pseudo_imagenet_folder as pseudo_imagenet_folder
from pcode.datasets.loader.svhn_folder import define_svhn_folder
from pcode.datasets.loader.femnist import define_femnist_folder
import pcode.utils.op_paths as op_paths

"""the entry for classification tasks."""


def _get_cifar(conf, name, root, split, transform, target_transform, download):
    is_train = split == "train"

    # decide normalize parameter.
    if name == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = (
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            if not conf.use_fake_centering
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
    elif name == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = (
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            if not conf.use_fake_centering
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
    normalize = normalize if conf.pn_normalize else None

    # decide data type.
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
            ]
            + ([normalize] if normalize is not None else [])
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
        )
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_cinic(conf, name, root, split, transform, target_transform, download):
    is_train = split == "train"

    # download dataset.
    if download:
        # create the dir.
        op_paths.build_dir(root, force=False)

        # check_integrity.
        is_valid_download = True
        for _type in ["train", "valid", "test"]:
            _path = os.path.join(root, _type)
            if len(os.listdir(_path)) == 10:
                num_files_per_folder = [
                    len(os.listdir(os.path.join(_path, _x))) for _x in os.listdir(_path)
                ]
                num_files_per_folder = [x == 9000 for x in num_files_per_folder]
                is_valid_download = is_valid_download and all(num_files_per_folder)
            else:
                is_valid_download = False

        if not is_valid_download:
            # download.
            torchvision.datasets.utils.download_and_extract_archive(
                url="https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz",
                download_root=root,
                filename="cinic-10.tar.gz",
                md5=None,
            )
        else:
            print("Files already downloaded and verified.")

    # decide normalize parameter.
    normalize = transforms.Normalize(
        mean=(0.47889522, 0.47227842, 0.43047404),
        std=(0.24205776, 0.23828046, 0.25874835),
    )
    normalize = normalize if conf.pn_normalize else None

    # decide data type.
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
            ]
            + ([normalize] if normalize is not None else [])
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
        )
    return torchvision.datasets.ImageFolder(root=root, transform=transform)


def _get_mnist(conf, root, split, transform, target_transform, download):
    is_train = split == "train"
    normalize = (
        transforms.Normalize((0.1307,), (0.3081,)) if conf.pn_normalize else None
    )

    transform = transforms.Compose(
        [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
    )
    return datasets.MNIST(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_stl10(conf, name, root, split, transform, target_transform, download):
    # right now this function is only used for unlabeled dataset.
    is_train = split == "train"

    # try to extract the downsample size if it has
    downsampled_size = conf.img_resolution

    # define the normalization operation.
    normalize = (
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if conf.pn_normalize
        else None
    )

    if is_train:
        split = "train+unlabeled"
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop((96, 96), 4)]
            + (
                [torchvision.transforms.Resize((downsampled_size, downsampled_size))]
                if downsampled_size is not None
                else []
            )
            + [transforms.ToTensor()]
            + ([normalize] if normalize is not None else [])
        )
    else:
        transform = transforms.Compose(
            (
                [torchvision.transforms.Resize((downsampled_size, downsampled_size))]
                if downsampled_size is not None
                else []
            )
            + [transforms.ToTensor()]
            + ([normalize] if normalize is not None else [])
        )
    return datasets.STL10(
        root=root,
        split=split,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_svhn(conf, root, split, transform, target_transform, download):
    is_train = split == "train"
    normalize = (
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if conf.pn_normalize
        else None
    )

    transform = transforms.Compose(
        [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
    )
    return define_svhn_folder(
        root=root,
        is_train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_femnist(conf, root, split, transform, target_transform, download):
    is_train = split == "train"
    assert (
        conf.pn_normalize is False
    ), "we've already normalize the image betwewen 0 and 1"

    transform = transforms.Compose([transforms.ToTensor()])
    return define_femnist_folder(
        root=root,
        is_train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_imagenet(conf, name, datasets_path, split):
    is_train = split == "train"
    is_downsampled = "8" in name or "16" in name or "32" in name or "64" in name
    root = os.path.join(
        datasets_path, "lmdb" if not is_downsampled else "downsampled_lmdb"
    )

    # get transform for is_downsampled=True.
    if is_downsampled:
        normalize = (
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            if conf.pn_normalize
            else None
        )

        if is_train:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
                + ([normalize] if normalize is not None else [])
            )
        else:
            transform = transforms.Compose(
                [transforms.ToTensor()] + ([normalize] if normalize is not None else [])
            )
    else:
        transform = None

    if conf.use_lmdb_data:
        if is_train:
            root = os.path.join(
                root, "{}train.lmdb".format(name + "_" if is_downsampled else "")
            )
        else:
            root = os.path.join(
                root, "{}val.lmdb".format(name + "_" if is_downsampled else "")
            )
        return imagenet_folder.define_imagenet_folder(
            conf=conf,
            name=name,
            root=root,
            flag=True,
            cuda=conf.graph.on_cuda,
            transform=transform,
            is_image=True and not is_downsampled,
        )
    else:
        return imagenet_folder.ImageNetDS(
            root=root, img_size=int(name[8:]), train=is_train, transform=transform
        )


def _get_pseudo_imagenet(conf, root, split="train"):
    is_train = split == "train"
    assert is_train

    # define normalize.
    normalize = (
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if conf.pn_normalize  # map to [-1, 1].
        else None
    )
    # define the transform.
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop((112, 112), 4)]
        + (
            [transforms.Resize((conf.img_resolution, conf.img_resolution))]
            if conf.img_resolution is not None
            else []
        )
        + [transforms.ToTensor()]
        + ([normalize] if normalize is not None else [])
    )
    # return the dataset.
    return pseudo_imagenet_folder.ImageNetDS(
        root=root, train=is_train, transform=transform
    )

def _get_plants(conf, root, split="train"):

    import json

    path = root + 'Preprocessed_dictionary_new_conf_modified_without_enh.json'

    with open(path, 'r') as f:
        disease_data = json.load(f)

    file_path =  root + "load.json"

    with open(file_path, "r") as d:
        dic = json.load(d)

    client_id = conf.client_id
    dataset = dic[str(client_id)]

    print("Selected dataset:", dataset)

    dataset = disease_data[dataset]

    train_loader, val_loader, test_loader = load_data(dataset, batch_size=conf.batch_size)

    if split == "train":
        return train_loader
    elif split == "val":
        return val_loader
    elif split == "test":
        return test_loader


"""the entry for different supported dataset."""


def get_dataset(
    conf,
    name,
    datasets_path,
    split="train",
    transform=None,
    target_transform=None,
    download=True,
):
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, name)

    if name == "cifar10" or name == "cifar100":
        return _get_cifar(
            conf, name, root, split, transform, target_transform, download
        )
    elif name == "cinic":
        return _get_cinic(
            conf, name, root, split, transform, target_transform, download
        )
    elif "stl10" in name:
        return _get_stl10(
            conf, name, root, split, transform, target_transform, download
        )
    elif name == "svhn":
        return _get_svhn(conf, root, split, transform, target_transform, download)
    elif name == "mnist":
        return _get_mnist(conf, root, split, transform, target_transform, download)
    elif name == "femnist":
        return _get_femnist(conf, root, split, transform, target_transform, download)
    elif "pseudo_imagenet" in name:
        return _get_pseudo_imagenet(conf, root, split)
    elif "imagenet" in name:
        return _get_imagenet(conf, name, datasets_path, split)
    elif name == "plants":
        cur_dir = os.path.dirname(__file__)
        print("Current directory:", cur_dir)
        root = "../../../../../../Flower_federated_bash/"
        return _get_plants(conf, root, split)
    else:
        raise NotImplementedError
    

if __name__ == "__main__":

    from torch.utils.data import Dataset
    
    def balance_data_from_json(
        json_data, train_ratio=0.7, val_ratio=0.1, random_seed=42
    ):  # max_samples_per_class=500):
        from sklearn.model_selection import train_test_split

        splits = {"train": [], "val": [], "test": []}

        for disease, v in json_data.items():
            for fruit, paths in v.items():

                # Controlla se ci sono immagini per questa classe
                if len(paths) == 0:
                    continue  # Salta questa classe se non ci sono immagini

                # Suddividi in train e temp (val + test)
                train_paths, temp_paths = train_test_split(
                    paths, train_size=train_ratio, random_state=random_seed
                )
                test_paths = temp_paths

                # Aggiungi i campioni ai rispettivi split
                splits["train"].extend(train_paths)
                splits["test"].extend(test_paths)

            print(
                f"{disease}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test"
            )
        print(
            f"Total: {len(splits['train']) + len(splits['val']) + len(splits['test'])} images"
        )
        return splits


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
            import torch
            # Ottieni il path del file
            file_path = self.file_paths[idx]

            label = None
            for class_name in self.class_labels:
                if f"/{class_name}/" in file_path:
                    label = self.label_map[class_name]
                    break
            if label is None:
                raise ValueError(f"Etichetta non trovata per il file: {file_path}")

            # Carica l'immagine dal file .npz
            data = np.load(file_path)
            image = data[
                "normalized_image"
            ]  # Supponiamo che l'immagine sia salvata con chiave 'image'

            # Converti immagine in tensore di PyTorch
            image = torch.tensor(image, dtype=torch.float32)

            return image, label
    
    def load_data(dataset, batch_size):
        """Load CIFAR-10 (training and test set)."""
        from torch.utils.data import DataLoader
        

        split_data = balance_data_from_json(dataset, val_ratio=0, random_seed=42)

        # Visualizza il numero di immagini in ogni split
        print(f"Train set: {len(split_data['train'])} immagini")
        print(f"Validation set: {len(split_data['val'])} immagini")
        print(f"Test set: {len(split_data['test'])} immagini \n")

        class_labels = list(dataset.keys())

        # Creazione dei Dataset
        train_dataset = CustomDataset(split_data["train"], class_labels)
        val_dataset = CustomDataset(split_data["val"], class_labels)
        test_dataset = CustomDataset(split_data["test"], class_labels)

        # Visualizza il primo elemento
        # image, label = train_dataset[0]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

