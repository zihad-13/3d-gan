# 3d-gan

## Training:
1. Prepare voxel data and keep them in similar structure as data folder. The 'data' folder is a sample of data directory structure. Here I have used ShapeNet dataset
with chair object.
2. If you have .obj data you can convert them into .binvox by ```gen_voxels.py``` 
3. Set proper parameters in ```config.py```.
4. To train use ```train_3D-GAN.py```

## Inference

provide the path of the generator model and run ```inference.py```

### Sample output
![MicrosoftTeams-image (4)](https://user-images.githubusercontent.com/38187481/220277291-77749f85-12f5-4844-8614-124667041022.png)
