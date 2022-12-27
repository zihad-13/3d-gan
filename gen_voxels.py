import os
import shutil
import glob
import platform

input_dir = "data/3d_shapes/chair"
output_dir = 'data/voxel'
n_models = None  # number of models to extract (None means all)
separator="/"
_len = len([f for f in os.path.split(input_dir) if f != ''])
dataset = os.path.join(output_dir, os.path.split(input_dir)[-1] if _len > 1 else input_dir)
# make chair model directory
if os.path.isdir(dataset):
    shutil.rmtree(dataset)
os.makedirs(dataset)

objs = [file for file in glob.glob('%s/*/models/*.obj'%input_dir)]
if n_models is None:
    n_models = len(objs)


for i in range(min([len(objs), n_models])):
    model_name = objs[i].split(separator)[-3]
    if platform.system().lower()=='linux':
        os.system(f"./binvox -d 64 -cb {objs[i]}")  
    elif platform.system().lower()=='windows':
        os.system(f"binvox.exe -d 64 -cb {objs[i]}")  

    os.makedirs(f"{dataset}/{model_name}/models/", exist_ok=True)
    shutil.move(f"{objs[i][:-4]}.binvox", f"{dataset}/{model_name}/models/{model_name}.binvox")
