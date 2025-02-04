import json
import os
from sklearn.model_selection import train_test_split

with open('Preprocessed_dictionary.json', 'r') as json_file:
    disease_data = json.load(json_file)

with open('Dictionary.json', 'r') as json_file:
    complete_dict = json.load(json_file)


list_dataset = ["PlantVillage", "Potato Leaf", "Potato Disease Leaf Dataset", "Dataset of Tomato Leaves", "PlantDoc", "Apple Tree Leaf Disease Segmentation Dataset", "Corn Leaf Disease", "Indigenous Dataset for Apple Leaf Disease Detection and Classification", "JMUBEN 3", "Rice Leaf Disease Image", "Sugarcane Leaf Image", "Sugarcane Leaf Disease"]
list_disease = ["Healthy", "Late_blight", "Early_blight", "Bacterial_spot", "Powdery_mildew", "Black_rot", "Rust", "Brown_spot", "Yellow", "Mosaic"]
fruit = ["Grape", "Potato", "Apple", "Tomato", "Peach", "Cherry", "Squash", "Pepper_bell", "Blueberry", "Corn", "Raspberry", "Strawberry","Soyabean", "Sugarcane", "Rice", "SweetPotato"]


def balance_data_from_json(json_data, target_datasets, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_seed=42): # max_samples_per_class=500):
    splits = {'train': [], 'val': [], 'test': []}
    
    for dataset, classes in json_data.items(): 
        if dataset not in target_datasets:
            continue 
         
        for _, v in classes.items(): 
            for _, paths in v.items():
               
                if len(paths) == 0:
                    continue  

                train_paths, temp_paths = train_test_split(
                    paths, 
                    train_size=train_ratio, 
                    random_state=random_seed
                )
                
                if val_ratio == 0:
                    val_paths = []  
                    test_paths = temp_paths
                elif test_ratio == 0:
                    test_paths = []  
                    val_paths = temp_paths
                else:
                    val_test_ratio = val_ratio / (val_ratio + test_ratio)
                    
                    val_paths, test_paths = train_test_split(
                        temp_paths, 
                        train_size=val_test_ratio, 
                        random_state=random_seed
                    )
                
                splits['train'].extend(train_paths)
                splits['val'].extend(val_paths)
                splits['test'].extend(test_paths)
    
    return splits



all_test_files = []


for target in list_dataset:
    target_datasets = [target]  
    split_data = balance_data_from_json(disease_data, target_datasets, val_ratio=0, random_seed=42)
    
    all_test_files.extend(split_data['test'])


final_dict = {dataset: {disease: [] for disease in list_disease} for dataset in list_dataset}

file_paths_test = all_test_files

for file_path in file_paths_test:
    for dataset in list_dataset:
        if dataset in file_path:
            for disease in list_disease:
                if disease in file_path:
                    final_dict[dataset][disease].append(file_path)
                    break  
            break  


total_file_paths = sum(len(paths) for dataset in final_dict.values() for paths in dataset.values())

print(json.dumps(final_dict, indent=4))

with open('test.json', 'w') as output_file:
    json.dump(final_dict, output_file, indent=4)


test_dict_jpg = {dataset: {disease: [] for disease in list_disease} for dataset in list_dataset}

for dataset, diseases in final_dict.items():
    for disease, paths in diseases.items():
        for path in paths:
            path_parts = path.split(os.sep)
            dataset_name = path_parts[2] 
            disease_name = path_parts[3] 
            fruit_name = path_parts[4]  
            base_name = os.path.splitext(os.path.basename(path))[0]  
            if dataset_name in complete_dict and disease_name in complete_dict[dataset_name]:
                if fruit_name in complete_dict[dataset_name][disease_name]:
                    images = complete_dict[dataset_name][disease_name][fruit_name]
                    
                    for img_path in images:
                        img_base_name = os.path.splitext(os.path.basename(img_path))[0]
                        
                        if base_name == img_base_name:
                            test_dict_jpg[dataset_name][disease_name].append(img_path)


print(json.dumps(test_dict_jpg, indent=4))



total_file_paths = sum(len(paths) for dataset in test_dict_jpg.values() for paths in dataset.values())


with open('test_jpg.json', 'w') as output_file:
    json.dump(test_dict_jpg, output_file, indent=4)

