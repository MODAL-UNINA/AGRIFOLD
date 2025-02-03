import os
import json


file_path = "Dictionary.json"

list_dataset = ["PlantVillage", "Potato Leaf", "Potato Disease Leaf Dataset", "Dataset of Tomato Leaves", "PlantDoc", "Apple Tree Leaf Disease Segmentation Dataset", "Corn Leaf Disease", "Indigenous Dataset for Apple Leaf Disease Detection and Classification", "JMUBEN 3", "Rice Leaf Disease Image", "Sugarcane Leaf Image", "Sugarcane Leaf Disease"]

list_Disease = ["Healthy", "Late_blight", "Early_blight", "Bacterial_spot", "Powdery_mildew", "Black_rot", "Rust", "Brown_spot", "Yellow", "Mosaic"]

Fruit = ["Grape", "Potato", "Apple", "Tomato", "Peach", "Cherry", "Squash", "Pepper_bell", "Blueberry", "Corn", "Raspberry", "Strawberry","Soyabean", "Sugarcane", "Rice", "SweetPotato"]

load_path = "Datasets" 


def func_Plant(d,j,f,load_path):
    l = []
    if os.path.isdir(os.path.join(load_path,d, j)): 
                    for name in os.listdir(os.path.join(load_path,d, j)): 
                        if f in name:
                            folder = name

                            if os.path.isdir(os.path.join(load_path,d, j, folder)):
                                l = sorted([os.path.join(load_path, d, j,folder, file) for file in os.listdir(os.path.join(load_path, d, j, folder))])
    return l




def func_dictionary(file_path, list_dataset, list_Disease, Fruit, load_path):
    
    dic = {}
    
    for d in list_dataset: 
        dic[d] = {}
        for j in list_Disease: 
            dic[d][j] = {}
            for f in Fruit: 
 
                if load_path == "Datasets":

                    if d == "PlantVillage":
                        dic[d][j][f] = func_Plant(d,j,f,load_path)
                          
                    if d == "Potato Leaf" or "Dataset of Tomato Leaves" or "Potato Disease Leaf Dataset":
                        if os.path.isdir(os.path.join(load_path,d, j)):
                        
                            l = sorted([os.path.join(load_path, d, j, file) for file in os.listdir(os.path.join(load_path, d, j))])
                            if f in d or (d == "Dataset of Tomato Leaves" and f == "Tomato") :
                                dic[d][j][f] = l

                    if d == "PlantDoc":
                            if os.path.isdir(os.path.join(load_path, d, f + "_" + j)):
                                l = sorted([os.path.join(load_path, d, f + "_" + j, file) for file in os.listdir(os.path.join(load_path, d, f + "_" + j))])
                                dic[d][j][f] = l

                    if d == "Apple Tree Leaf Disease Segmentation Dataset":
                        if os.path.isdir(os.path.join(load_path,d, j)):
                        
                            l = sorted([os.path.join(load_path, d, j, file) for file in os.listdir(os.path.join(load_path, d, j))])
                            if f in d :
                                dic[d][j][f] = l

                    if d == "Corn Leaf Disease":
                        if os.path.isdir(os.path.join(load_path,d, j)):
                        
                            l = sorted([os.path.join(load_path, d, j, file) for file in os.listdir(os.path.join(load_path, d, j))])
                            if f in d :
                                dic[d][j][f] = l
                    
                    if d == "Indigenous Dataset for Apple Leaf Disease Detection and Classification":
                        if os.path.isdir(os.path.join(load_path,d, j)):
                        
                            l = sorted([os.path.join(load_path, d, j, file) for file in os.listdir(os.path.join(load_path, d, j))])
                            if f in d :
                                dic[d][j][f] = l

                    if d == "JMUBEN 3":
                        if os.path.isdir(os.path.join(load_path,d, j)):
                        
                            l = sorted([os.path.join(load_path, d, j, file) for file in os.listdir(os.path.join(load_path, d, j))])
                            if (d == "JMUBEN 3" and f == "SweetPotato") :
                                dic[d][j][f] = l


                    if d == "Rice Leaf Disease Image":
                        if os.path.isdir(os.path.join(load_path,d, j)):
                        
                            l = sorted([os.path.join(load_path, d, j, file) for file in os.listdir(os.path.join(load_path, d, j))])
                            if f in d :
                                dic[d][j][f] = l

                    if d == "Sugarcane Leaf Image":
                        if os.path.isdir(os.path.join(load_path,d, j)):
                        
                            l = sorted([os.path.join(load_path, d, j, file) for file in os.listdir(os.path.join(load_path, d, j))])
                            if f in d :
                                dic[d][j][f] = l

                    if d == "Sugarcane Leaf Disease":
                        if os.path.isdir(os.path.join(load_path,d, j)):
                        
                            l = sorted([os.path.join(load_path, d, j, file) for file in os.listdir(os.path.join(load_path, d, j))])
                            if f in d :
                                dic[d][j][f] = l



                else:
                    dic[d][j][f] = func_Plant(d,j,f,load_path)



    with open(file_path, 'w') as json_file:
        json.dump(dic, json_file)



    return dic
                

# Dictionary creation
dic = func_dictionary(file_path, list_dataset, list_Disease, Fruit, load_path)

                                
