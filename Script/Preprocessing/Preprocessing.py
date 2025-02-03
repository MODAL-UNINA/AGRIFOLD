import cv2
import json
import os
import numpy as np
import tqdm as tqdm

  
file_path_1 = "Dictionary.json"

output_folder = "Preprocessed_datasets"

list_dataset = ["PlantVillage", "Potato Leaf", "Potato Disease Leaf Dataset", "Dataset of Tomato Leaves", "PlantDoc", "Apple Tree Leaf Disease Segmentation Dataset", "Corn Leaf Disease", "Indigenous Dataset for Apple Leaf Disease Detection and Classification", "JMUBEN 3", "Rice Leaf Disease Image", "Sugarcane Leaf Image", "Sugarcane Leaf Disease"]

list_disease = ["Healthy", "Late_blight", "Early_blight", "Bacterial_spot", "Powdery_mildew", "Black_rot", "Rust", "Brown_spot", "Yellow", "Mosaic"]

fruit = ["Grape", "Potato", "Apple", "Tomato", "Peach", "Cherry", "Squash", "Pepper_bell", "Blueberry", "Corn", "Raspberry", "Strawberry","Soyabean", "Sugarcane", "Rice", "SweetPotato"]

load_path = "Preprocessed_datasets" 

file_path_2 = "Preprocessed_dictionary.json"



def load_data(dataset, file_path, output_folder):

    input_shape = (224,224,3)
   
    with open(file_path, 'r') as d:
        dic = json.load(d)

    dic = dic[dataset] 
  
    dataset_output_folder = os.path.join(output_folder, dataset)
    os.makedirs(dataset_output_folder, exist_ok=True)

    for disease in dic.keys(): 
        for fruit in dic[disease].keys(): 
            for img_path in dic[disease][fruit]: 
                image = cv2.imread(img_path)
                
                h = image.shape[0]
                w = image.shape[1]

                # Make border
                if w != h:
                    if w < h:
                        x = (h - w)//2
                        image = cv2.copyMakeBorder(image, 0, 0, x, x, cv2.BORDER_CONSTANT, None, value = (255, 255, 255)) 
                    else:
                        x = (w - h)//2
                        image = cv2.copyMakeBorder(image, x, x, 0, 0, cv2.BORDER_CONSTANT, None, value = (255, 255, 255))

                # Resizing
                if image.shape != input_shape:

                    resized_image = cv2.resize(src= image, 
                        dsize =input_shape[:2], 
                        interpolation=cv2.INTER_LANCZOS4)

                        
                assert resized_image.shape[:2] == (224, 224), f"The resized image is not 224x224, but {resized_image.shape[:2]}"

                # Normalization    
                b, g, r = cv2.split(resized_image)

                # Normalization parameters
                min_value = 0.
                max_value = 1.
                norm_type = cv2.NORM_MINMAX

                    
                b_normalized = cv2.normalize(b.astype('float'), None, min_value, max_value, norm_type)
                g_normalized = cv2.normalize(g.astype('float'), None, min_value, max_value, norm_type)
                r_normalized = cv2.normalize(r.astype('float'), None, min_value, max_value, norm_type)
 
                normalized_image = cv2.merge((b_normalized, g_normalized, r_normalized))
    
                base_name = os.path.splitext(os.path.basename(img_path))[0]
              
                output_disease_folder = os.path.join(dataset_output_folder, disease)
                output_fruit_folder = os.path.join(output_disease_folder, fruit)
             
                os.makedirs(output_fruit_folder, exist_ok=True)
        
                output_file = os.path.join(output_fruit_folder, f"{base_name}.npz")

                np.savez_compressed(output_file, normalized_image=normalized_image)



# Preprocessing
for d in tqdm.tqdm(list_dataset, total = len(list_dataset)):
    load_data(d, file_path_1, output_folder)

# Dictionary creation
from Dictionary_creation import func_dictionary
dic = func_dictionary(file_path_2, list_dataset, list_disease, fruit, load_path)            
                








