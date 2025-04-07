# %%
import json
import os
import time
import torch
import argparse
import flwr as fl
import numpy as np
import torchprofile
from tqdm import tqdm
import torch.nn as nn
import torch_pruning as tp
from torchvision import models
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from collections import OrderedDict
from flwr.common.typing import Scalar
from torch.utils.data import DataLoader, Dataset

from utils import is_interactive
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument("--gpuid", type=int, default=1, help="GPU number to use")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
parser.add_argument("--client_id", type=int, default=0, help="Client ID")
parser.add_argument("--epochs", type=int, default=1, help="Epochs per round")
parser.add_argument("--server-address", type=str, default="localhost:12389", help="Address of the server")
parser.add_argument("--ablation", type=str, default="None", help="Ablation study")
parser.add_argument("--pruning", type=bool, default=False, help="Pruning")
parser.add_argument("--pruning-ratio", type=float, default=0.5, help="Pruning ratio")
parser.add_argument("--bn", type=bool, default=False, help="Set to True if you want to use FedBN model")

if is_interactive():
  args, _ = parser.parse_known_args()
else:
  args = parser.parse_args()

bn = args.bn
gpu_id = args.gpuid
batch_size = args.batch_size
num_epochs = args.epochs
client_id = args.client_id
server_address = args.server_address

DEVICE = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

  
class Net(nn.Module):
  """Constructs a ECA module.

  Args:
    channel: Number of channels of the input feature map
    k_size: Adaptive selection of kernel size
  """
  def __init__(self, channel, k_size=3):
    super(Net, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
    self.sigmoid = nn.Sigmoid()
    self.feature_maps = None
      

  def forward(self, x):

    y = self.avg_pool(x)

    # Two different branches of ECA module
    y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

    # Multi-scale information fusion
    y = self.sigmoid(y)

    self.feature_maps = x * y.expand_as(x)

    return self.feature_maps 


class Net_bn(nn.Module):
    """Constructs a ECA module.

    Args:
      channel: Number of channels of the input feature map
      k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(Net_bn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        # batch norm
        self.bn = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Sigmoid()
        self.feature_maps = None

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        self.feature_maps = x * y.expand_as(x)

        return self.feature_maps

class VGG16WithECA(nn.Module):
  def __init__(self, num_classes=10, kernel_size=3, pretrained=True):
    super(VGG16WithECA, self).__init__()

    vgg = models.vgg16(pretrained=pretrained)

    for param in vgg.features.parameters():
      param.requires_grad = True
    
    self.features = nn.Sequential(

      *vgg.features[:5],  # Conv1-Conv2 + ReLU
      Net(kernel_size),     

      *vgg.features[5:10],  # Conv3-Conv4 + ReLU
      Net(kernel_size),

      *vgg.features[10:17],  # Conv5-Conv7 + ReLU
      Net(kernel_size),
      
      *vgg.features[17:24],  # Conv8-Conv10 + ReLU
      Net(kernel_size),

      *vgg.features[24:],  # Conv11-Conv13 + ReLU
      Net(kernel_size)
    )

    if args.ablation == 'a':
      self.classifier = nn.Sequential(
          nn.Linear(512 * 7 * 7, 2048),  
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(2048, num_classes)  
      )
    
    elif args.ablation == 'b':
      self.classifier = nn.Linear(512 * 7 * 7, num_classes)

    elif args.ablation == 'c':

      self.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 1024),  
        nn.ReLU(),                     
        nn.Dropout(0.5),               
        nn.Linear(1024, num_classes)   
        )
    
    else:
      self.classifier = vgg.classifier
      self.classifier[-1] = nn.Linear(4096, num_classes)
      

    
  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x
  
  def get_last_eca_feature_maps(self):
    return self.features[-1].feature_maps

class VGG16WithECA_bn(nn.Module):
    def __init__(self, num_classes=10, kernel_size=3, pretrained=True):
        super(VGG16WithECA_bn, self).__init__()

        vgg = models.vgg16(pretrained=pretrained)

        for param in vgg.features.parameters():
            param.requires_grad = True

        self.features = nn.Sequential(

            *vgg.features[:5],  # Conv1-Conv2 + ReLU
            Net_bn(kernel_size),

            *vgg.features[5:10],  # Conv3-Conv4 + ReLU
            Net_bn(kernel_size),

            *vgg.features[10:17],  # Conv5-Conv7 + ReLU
            Net_bn(kernel_size),

            *vgg.features[17:24],  # Conv8-Conv10 + ReLU
            Net_bn(kernel_size),

            *vgg.features[24:],  # Conv11-Conv13 + ReLU
            Net_bn(kernel_size),
        )

        self.classifier = vgg.classifier
        self.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_last_eca_feature_maps(self):
        return self.features[-1].feature_maps

class CustomDataset(Dataset):
    def __init__(self, file_paths, class_labels, transform=None):
      
      self.file_paths = file_paths
      self.class_labels = class_labels
      self.transform = transform

      self.label_map = {label: idx for idx, label in enumerate(self.class_labels)}

    def __len__(self):
      return len(self.file_paths)

    def __getitem__(self, idx):
      file_path = self.file_paths[idx]

      label = None
      for class_name in self.class_labels:
        if f"/{class_name}/" in file_path:
          label = self.label_map[class_name]
          break
      if label is None:
        raise ValueError(f"Label not found for file: {file_path}")

      data = np.load(file_path)
      image = data['normalized_image']

      image = torch.tensor(image, dtype=torch.float32)

      return image, label
      

def train(client_id, net, train_loader, val_loader, epochs, train_losses, train_accuracies, val_losses, val_accuracies, proximal_mu):
  
  """Train the model on the training set."""
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

  global_params = [p.clone().detach() for p in net.parameters()]

  for epoch in range(epochs): 
    net.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    pbar = tqdm(train_loader, total=len(train_loader))
  
    for images, labels in pbar: 
      labels = labels.to(DEVICE)
      optimizer.zero_grad()
      outputs = net(images.permute(0, 3, 1, 2).to(DEVICE))

      # FedProx --> FedAvg if proximal_mu == 0
      proximal_term = 0
      for i, (local_weights, global_weights) in enumerate(zip(net.parameters(), global_params)):
          proximal_term += (local_weights - global_weights).norm(2)
      loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term

      loss.backward()
      optimizer.step()

      train_loss += loss.item() * labels.size(0)
      _, predicted = torch.max(outputs.data, 1)
      total_train += labels.size(0)
      correct_train += (predicted == labels).sum().item()
      pbar.set_postfix({'Batch Loss': loss.item(), 'Loss': train_loss / total_train, 'Accuracy': (100 * correct_train / total_train)})

    avg_train_loss = train_loss / total_train
    train_accuracy = 100 * correct_train / total_train

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)


    print ('Epoch [{}/{}], Training Loss: {:.4f}, Accuracy: {:.2f}%' 
          .format(epoch+1, epochs, avg_train_loss, train_accuracy))
    

    model_weights = net.state_dict()
    torch.save(model_weights, f"model_eca_client_{client_id}.pth")


    if len(val_loader.dataset) != 0:
                
      # Validation
      net.eval()
      val_loss = 0
      correct_val = 0
      total_val = 0
      with torch.no_grad():
        for images, labels in val_loader: 
          labels = labels.to(DEVICE)
          images = images.permute(0,3,1,2).to(DEVICE) 
          outputs = net(images).to(DEVICE)
          val_loss += criterion(outputs, labels).item() * labels.size(0)
          _, predicted = torch.max(outputs.data, 1)
          total_val += labels.size(0)
          correct_val += (predicted == labels).sum().item()
          #del images, labels, outputs
          
          avg_val_loss = val_loss / total_val
          val_accuracy = 100 * correct_val / total_val

          val_losses.append(avg_val_loss)
          val_accuracies.append(val_accuracy)

          print('Epoch [{}/{}], Validation Loss: {:.4f}, Accuracy: {:.2f}%'.format(
          epoch + 1, epochs, avg_val_loss, val_accuracy))

           # Early stopping
          if val_loss < best_loss:
            best_loss = val_loss
            patience = 10
          else:
            patience -= 1
            if patience == 0:
              print("Early stopping triggered!")
              break

  return  train_losses, train_accuracies, val_losses, val_accuracies


def test(net, testloader, class_labels):
    """Validate the model on the test set."""

    all_labels = []
    all_predictions = []

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            labels = labels.to(DEVICE)
            images = images.to(DEVICE)
            images = images.permute(0, 3, 1, 2)
            outputs = net(images)
            loss += criterion(outputs, labels.to(DEVICE)).item()

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Accuracy of the network on the {} test images: {} %".format(
                len(testloader.dataset), 100 * correct / total
            )
        )

        print(
            "Loss of the network on the {} test images: {}".format(
                len(testloader.dataset), loss / len(testloader.dataset)
            )
        )

        print("\nClassification Report:")
        labels = list(range(len(class_labels)))
        print(classification_report(all_labels, all_predictions, labels=labels, target_names=class_labels))


    return loss / len(testloader.dataset), correct / total




def balance_data_from_json(json_data, train_ratio=0.7, random_seed=42):
  
  splits = {'train': [], 'val': [], 'test': []}

  for disease, v in json_data.items(): 
    for fruit, paths in v.items(): 

      if len(paths) == 0:
        continue

      train_paths, temp_paths = train_test_split(
        paths, 
        train_size=train_ratio, 
        random_state=random_seed
      )
      test_paths = temp_paths

      splits['train'].extend(train_paths)
      splits['test'].extend(test_paths)

  print(f"{disease}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
  print(f"Total: {len(splits['train']) + len(splits['val']) + len(splits['test'])} images")
  return splits


def load_data(dataset, batch_size):

  split_data = balance_data_from_json(dataset, random_seed=42)


  print(f"Train set: {len(split_data['train'])} images")
  print(f"Validation set: {len(split_data['val'])} images")
  print(f"Test set: {len(split_data['test'])} images")

  class_labels = list(dataset.keys())

  # Dataset
  train_dataset = CustomDataset(split_data['train'], class_labels)
  val_dataset = CustomDataset(split_data['val'], class_labels)
  test_dataset = CustomDataset(split_data['test'], class_labels)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader, test_loader



# %%
class FlowerClient(fl.client.NumPyClient):
  
  def __init__(self, client_id) -> None:
        
    self.train_losses = []
    self.train_accuracies = []
    self.val_losses = []
    self.val_accuracies = []
    self.all_labels = []
    self.all_predictions = []
    self.client_id = client_id
    self.num_epochs = num_epochs 
    self.bn = bn

  def get_parameters(self, config):
    if self.bn:
      return [
        val.cpu().numpy()
        for name, val in self.net.state_dict().items()
        if "bn" not in name
      ]
    else:
      return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

  def set_parameters(self, parameters):
    if self.bn:
      keys = [k for k in self.net.state_dict().keys() if "bn" not in k]
      params_dict = zip(keys, parameters)
      state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
      self.net.load_state_dict(state_dict, strict=False)
    else:
      params_dict = zip(self.net.state_dict().keys(), parameters)
      state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
      self.net.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):

    self.proximal_mu = config["proximal_mu"]

    self.set_parameters(parameters)
    self.train_losses , self.train_accuracies, self.val_losses, self.val_accuracies = train(
      self.client_id, net, train_loader, val_loader, self.num_epochs, self.train_losses, self.train_accuracies,
        self.val_losses, self.val_accuracies, self.proximal_mu)
    
    self.plot()

    return self.get_parameters(config={}), len(train_loader.dataset), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    loss, accuracy = test(net, test_loader)
    return float(loss), len(test_loader.dataset), {"accuracy": float(accuracy)}
  
  def plot(self):
     
    num_rounds = len(self.train_losses) // self.num_epochs

    round_ticks = [i * num_epochs for i in range(num_rounds)]

    output_dir = "output_flower"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'metrics_client_{client_id}.png')

    # Plot
    plt.figure(figsize=(12, 6))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(self.train_losses, label='Training Loss')
    if self.val_losses != []:
      plt.plot(self.val_losses, label='Validation Loss')
    plt.title('Loss Over Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.xticks(round_ticks, labels=[str(i + 1) for i in range(num_rounds)]) 
    plt.legend()

    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(self.train_accuracies, label='Training Accuracy')
    if self.val_accuracies != []:
      plt.plot(self.val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Over Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy (%)')
    plt.xticks(round_ticks, labels=[str(i + 1) for i in range(num_rounds)]) 
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file)


def calculate_model_size(model, file_path="temp_model.pth"):
    torch.save(model.state_dict(), file_path)
    model_size = os.path.getsize(file_path) / 1e6  # MB
    os.remove(file_path)
    return model_size


# %% SCAFFOLD

from torch.optim import SGD
class ScaffoldOptimizer(SGD):
  """Implements SGD optimizer step function as defined in the SCAFFOLD paper."""

  def __init__(self, grads, step_size, momentum, weight_decay):
    super().__init__(
      grads, lr=step_size, momentum=momentum, weight_decay=weight_decay
    )

  def step_custom(self, server_cv, client_cv, device):
    """Implement the custom step function fo SCAFFOLD."""
    self.step()
    for group in self.param_groups:
      for par, s_cv, c_cv in zip(group["params"], server_cv, client_cv):
        s_cv = s_cv.to(device)
        c_cv = c_cv.to(device)
        par.data.add_(s_cv - c_cv, alpha=-group["lr"])


def train_scaffold(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    server_cv: torch.Tensor,
    client_cv: torch.Tensor,
) -> None:
    """Train the network on the training set using SCAFFOLD.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The training set dataloader object.
    device : torch.device
        The device on which to train the network.
    epochs : int
        The number of epochs to train the network.
    learning_rate : float
        The learning rate.
    momentum : float
        The momentum for SGD optimizer.
    weight_decay : float
        The weight decay for SGD optimizer.
    server_cv : torch.Tensor
        The server's control variate.
    client_cv : torch.Tensor
        The client's control variate.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = ScaffoldOptimizer(
        net.parameters(), learning_rate, momentum, weight_decay
    )
    net.train()
    for epoch in range(epochs):
        net, train_loss, correct_train, total_train = _train_one_epoch_scaffold(
            net, trainloader, device, criterion, optimizer, server_cv, client_cv
        )

        avg_train_loss = train_loss / total_train
        train_accuracy = 100 * correct_train / total_train


        print(
            "Epoch [{}/{}], Training Loss: {:.4f}, Accuracy: {:.2f}%".format(
                epoch + 1, epochs, avg_train_loss, train_accuracy
            )
        )



def _train_one_epoch_scaffold(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: ScaffoldOptimizer,
    server_cv: torch.Tensor,
    client_cv: torch.Tensor,
) -> nn.Module:
    
    """Train the network on the training set for one epoch."""
    pbar = tqdm(trainloader, total=len(trainloader))
    train_loss = 0
    correct_train = 0
    total_train = 0
    for data, target in pbar:
        data, target = data.permute(0, 3, 1, 2).to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step_custom(server_cv, client_cv, device)

        train_loss += loss.item() * target.size(0)
        _, predicted = torch.max(output.data, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()

    return net, train_loss, correct_train, total_train


def test_scaffold(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to evaluate.
    testloader : DataLoader
        The test set dataloader object.
    device : torch.device
        The device on which to evaluate the network.

    Returns
    -------
    Tuple[float, float]
        The loss and accuracy of the network on the test set.
    """
    # criterion = nn.CrossEntropyLoss(reduction="sum")
    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    correct, total, loss = 0, 0, 0.0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        print("test loader length:", len(testloader))
        pbar = tqdm(testloader, total=len(testloader))
        for data, target in pbar:
            data, target = data.permute(0, 3, 1, 2).to(device), target.to(device)
            output = net(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            all_labels.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print(
            "Accuracy of the network on the {} test images: {} %".format(
                len(testloader.dataset), 100 * correct / total
            )
        )

        print(
            "Loss of the network on the {} test images: {}".format(
                len(testloader.dataset), loss / len(testloader.dataset)
            )
        )

    print("\nClassification Report:")
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(classification_report(all_labels, all_predictions, labels=labels, target_names=class_labels))

    return  loss / total, correct / total

class ScaffoldClient(fl.client.NumPyClient):
    """Flower client for SCAFFOLD."""
    def __init__(
        self,
        cid: int,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        save_dir: str = "",
    ) -> None:
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        # initialize client control variate with 0 and shape of the network parameters
        self.client_cv = []
        for param in self.net.parameters():
            self.client_cv.append(torch.zeros(param.shape))
        # save cv to directory
        if save_dir == "":
            save_dir = "client_cvs"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        print(f"Client {self.cid}: inviando parametri personalizzati!")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit_scaffold(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for SCAFFOLD."""
        # the first half are model parameters and the second are the server_cv
        server_cv = parameters[len(parameters) // 2 :]
        parameters = parameters[: len(parameters) // 2]
        self.set_parameters(parameters)
        self.client_cv = []
        for param in self.net.parameters():
            self.client_cv.append(param.clone().detach())
        # load client control variate
        if os.path.exists(f"{self.dir}/client_cv_{self.cid}.pt"):
            self.client_cv = torch.load(f"{self.dir}/client_cv_{self.cid}.pt")
        # convert the server control variate to a list of tensors
        server_cv = [torch.Tensor(cv) for cv in server_cv]
        train_scaffold(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
            server_cv,
            self.client_cv,
        )
        x = parameters
        y_i = self.get_parameters(config={})
        c_i_n = []
        server_update_x = []
        server_update_c = []
        # update client control variate
        for c_i_j, c_j, x_j, y_i_j in zip(self.client_cv, server_cv, x, y_i):
            c_i_n.append(
                c_i_j
                - c_j
                + (1.0 / (self.learning_rate * self.num_epochs * len(self.trainloader)))
                * (x_j - y_i_j)
            )

            server_update_x.append((y_i_j - x_j))
            server_update_c.append((c_i_n[-1] - c_i_j).cpu().numpy())
        self.client_cv = c_i_n
        torch.save(self.client_cv, f"{self.dir}/client_cv_{self.cid}.pt")

        combined_updates = server_update_x + server_update_c

        self.plot()

        return (
            combined_updates,
            len(self.trainloader.dataset),
            {},
        )
    
    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for SCAFFOLD."""

        server_cv = parameters[len(parameters) // 2 :]
        parameters = parameters[: len(parameters) // 2]
        self.set_parameters(parameters)
        self.client_cv = []
        for param in self.net.parameters():
            self.client_cv.append(param.clone().detach())
        # load client control variate
        if os.path.exists(f"{self.dir}/client_cv_{self.cid}.pt"):
            self.client_cv = torch.load(f"{self.dir}/client_cv_{self.cid}.pt")
        # convert the server control variate to a list of tensors
        server_cv = [torch.Tensor(cv) for cv in server_cv]
        train_scaffold(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
            server_cv,
            self.client_cv,
        )
        x = parameters
        y_i = self.get_parameters(config={})
        c_i_n = []
        server_update_x = []
        server_update_c = []
        # update client control variate c_i_1 = c_i - c + 1/eta*K (x - y_i)
        for c_i_j, c_j, x_j, y_i_j in zip(self.client_cv, server_cv, x, y_i):
            c_i_j = c_i_j.to('cpu')
            c_j = c_j.to('cpu')
            # x_j = x_j.to(self.device)
            # y_i_j = y_i_j.to(self.device)
            c_i_n.append(
                c_i_j
                - c_j
                + (1.0 / (self.learning_rate * self.num_epochs * len(self.trainloader)))
                * (x_j - y_i_j)
            )
            # y_i - x, c_i_n - c_i for the server
            server_update_x.append((y_i_j - x_j))
            server_update_c.append((c_i_n[-1] - c_i_j).cpu().numpy())
        self.client_cv = c_i_n
        torch.save(self.client_cv, f"{self.dir}/client_cv_{self.cid}.pt")

        combined_updates = server_update_x + server_update_c

        return (
            combined_updates,
            len(self.trainloader.dataset),
            {},
        )
    
    # def to_client(self):
    #     """Registra i metodi personalizzati e forza l'uso di questa classe."""
    #     return fl.client.Client(
    #         get_parameters=self.get_parameters,  # Registra esplicitamente la funzione
    #         fit=self.fit,
    #         evaluate=self.evaluate,
    #     )

    # def to_client(self):
    #     """Crea un client compatibile con Flower registrando i metodi personalizzati."""
    #     return fl.client.ClientApp(self) 

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        loss, acc = test_scaffold(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}
    

    def plot(self):

        # Calcolo del numero di round
        num_rounds = len(self.train_losses) // self.num_epochs

        # Creazione degli xticks: un tick per ogni round
        round_ticks = [i * num_epochs for i in range(num_rounds)]

        output_dir = "output_flower"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"metrics_client_{client_id}.png")

        # Plot delle metriche alla fine del training
        plt.figure(figsize=(12, 6))

        # Subplot 1: Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Training Loss")
        if self.val_losses != []:
            plt.plot(self.val_losses, label="Validation Loss")
        plt.title("Loss Over Rounds")
        plt.xlabel("Rounds")
        plt.ylabel("Loss")
        plt.xticks(round_ticks, labels=[str(i + 1) for i in range(num_rounds)])
        plt.legend()

        # Subplot 2: Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label="Training Accuracy")
        if self.val_accuracies != []:
            plt.plot(self.val_accuracies, label="Validation Accuracy")
        plt.title("Accuracy Over Rounds")
        plt.xlabel("Rounds")
        plt.ylabel("Accuracy (%)")
        plt.xticks(round_ticks, labels=[str(i + 1) for i in range(num_rounds)])
        plt.legend()

        # Salva il plot
        plt.tight_layout()
        plt.savefig(output_file)
        # plt.show()
    

# %%
file_path = "load.json"

with open(file_path, 'r') as d:
  dic = json.load(d)

disease_data_path = "Preprocessing/Preprocessed_dictionary.json"

with open(disease_data_path, 'r') as f:
  disease_data = json.load(f)


if client_id >= len(disease_data.keys()):
  import sys

  print(f"Client ID {client_id} does not exist for id {client_id}. Closing.")
  sys.exit(0)

print("Client ID:", client_id)
dataset = dic[str(client_id)]

print("Selected dataset:" , dataset)

dataset = disease_data[dataset] 

sum_images = 0
for disease, v in dataset.items():
  for fruit, paths in v.items():
    sum_images += len(paths)
print(f"Total: {sum_images} images")

class_labels = list(dataset.keys())

# Load model and data
net = VGG16WithECA(num_classes = len(class_labels)).to(DEVICE)
train_loader, val_loader, test_loader = load_data(dataset, batch_size)

model_size = calculate_model_size(net)
print(f"Model size: {model_size:.2f} MB")

# Wait for server status file
server_status_file = f"../0/status_server.txt"
print('path exist:', os.path.exists(server_status_file))
while not os.path.exists(server_status_file):
  print(f"Waiting for server status file {server_status_file}...")
  time.sleep(5)

if args.pruning:
  example_inputs = torch.randn(1, 3, 224, 224).to(DEVICE)

  # 1. Importance criterion, here we calculate the L2 Norm of grouped weights as the importance score
  imp = tp.importance.GroupNormImportance(p=2) 

  # 2. Initialize a pruner with the model and the importance criterion
  ignored_layers = []
  for m in net.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
      ignored_layers.append(m) # DO NOT prune the final classifier!

  pruner = tp.pruner.MetaPruner(
    net,
    example_inputs,
    importance=imp,
    pruning_ratio=args.pruning_ratio,
    ignored_layers=ignored_layers,
    round_to=8,
  )

  # 3. Prune the model
  pruner.step()


  # 4. Compute MACs and num_params
  with torch.no_grad():
    profile = torchprofile.profile_macs(net, example_inputs)
    params = sum(p.numel() for p in net.parameters())
    print(f"MACs: {profile / 1e9} G, Params: {params / 1e6} M")

  model_size = calculate_model_size(net)
  print(f"Model size after pruning: {model_size:.2f} MB")

# Start Flower client -- choose the client

# FedAvg and FedProx
client=FlowerClient(client_id, net, bn).to_client()

# SCAFFOLD
# client=ScaffoldClient(client_id, net, train_loader, test_loader, DEVICE, num_epochs, 0.0001, 0.9, 0).to_client(),

fl.client.start_client(grpc_max_message_length = 1_074_422_052, server_address = server_address, client=client)

