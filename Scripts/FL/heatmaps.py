#%%
import os
import json
import torch
import cv2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models


class_labels = ["Healthy", "Late_blight", "Early_blight", "Bacterial_spot", "Powdery_mildew", "Black_rot", "Rust", "Brown_spot", "Yellow", "Mosaic"]


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

        # Classifier 
        self.classifier = vgg.classifier
        self.classifier[-1] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_last_eca_feature_maps(self):
        return self.features[-1].feature_maps


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = VGG16WithECA(num_classes=10, kernel_size=3, pretrained=True).to(device)


def process_image(npz_path, model):

    data = np.load(npz_path)
    images = data['normalized_image']

    input_tensor = torch.tensor(images, dtype=torch.float32).to(device)
    input_tensor = input_tensor.unsqueeze(dim=0).permute(0, 3, 1, 2)

    _ = model(input_tensor)
    last_eca_feature_maps = model.get_last_eca_feature_maps()

    # Heatmap
    dim_avg = torch.mean(last_eca_feature_maps, dim=1).squeeze()
    heatmap = dim_avg.cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = torch.tensor(heatmap)
    heatmap /= torch.max(heatmap)  

    heatmap = np.array(heatmap)
    images = (images * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (images.shape[1], images.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = heatmap_colored * 0.4 + images

    return images, superimposed_img

def predict_class(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


#%%

file_path_1 = "Preprocessing/test_data_jpg.json"


with open(file_path_1, 'r') as f:
    image_dict = json.load(f)
num_rows = len(image_dict)
num_cols = 2


for client_id in range(0, 12):
    model_path = f"model_eca_client_{client_id}.pth"
    
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        )
    )

    file_path_2 = f"test_data_client_{client_id}.json"

    with open(file_path_2, 'r') as f:
        preprocessed_dict = json.load(f)

    dataset = list(image_dict.keys())[client_id]
    classes = list(image_dict[dataset].keys())

    for class_name in classes:
        if preprocessed_dict[class_name] == []:
            print(f"{dataset}: Empty class: {class_name}")
            continue

        image = image_dict[dataset][class_name][1]
        npz = preprocessed_dict[class_name][1]

        original_img, heatmap_img = process_image(image, npz, model)


        data = np.load(npz)
        images = data['normalized_image']
        input_tensor = torch.tensor(images, dtype=torch.float32).to(device)
        input_tensor = input_tensor.unsqueeze(dim=0).permute(0, 3, 1, 2)

        predicted_index = predict_class(model, input_tensor)
        true_index = model.class_to_idx[class_name]
        prediction_correct = 1 if predicted_index == true_index else 0

        output_path = f'heatmaps/{dataset}_{class_name}.jpg'
        os.makedirs('heatmaps', exist_ok=True)
        cv2.imwrite(output_path, heatmap_img)

        heatmap_img = cv2.imread(output_path)
        original_img = cv2.imread(image)

        images = (images * 255).astype(np.uint8)
        original_img = images

        img1 = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img1)
        axes[0].axis('off')

        axes[1].imshow(img2)
        axes[1].axis('off')

        if dataset == 'Indigenous Dataset for Apple Leaf Disease Detection and Classification':
            fig.suptitle(f"Indigenous Dataset for Apple Leaf Disease \n Detection and Classification - {class_name}", fontsize=20)
        else:
            fig.suptitle(f"{dataset}\n{class_name}", fontsize=20)

        plt.subplots_adjust(top=0.9)
        plt.savefig(f"heatmaps/{dataset}_{class_name}_comparison.jpg")



