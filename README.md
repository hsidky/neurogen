# Neurogen

Neurogen is a library for converting data into compatible formats with Neuroglancer.
The two types of outputs it can create are:
1) Volume generation
2) Mutli-resolutional meshes



**Installation**

```
pip intall neurogen --recursive-submodules
```

To set up local environment:
https://github.com/google/neuroglancer#building

**Volume Generation Example**


Download example data and save in appropriate directory:
http://graphics.stanford.edu/data/voldata/cthead-8bit.tar.gz

More details of data: http://graphics.stanford.edu/data/voldata/

```python
import numpy as np
import os
from PIL import Image
import json
from neurogen import volume as ngvolume
from neurogen import mesh as ngmesh


# Unzip tar file into appropriate directory
input = "/home/input"
output = "/home/output"

#  Generate Input
volume = np.zeros((256,256,113,1,1)).astype('uint8') 
for filename in os.listdir(input):
    index = int(filename[11:14])
    filename = os.path.join(input,filename)
    image = Image.open(filename)
    imarray = np.array(image)
    volume[:,:,index,0,0] = imarray.astype('uint8')

# Generate the info specification file with the appropriate data
info = {
    "@type": "neuroglancer_multiscale_volume",
    "data_type": "uint8",
    "num_channels": 1,
    "scales": [
        {"chunk_sizes": [[256, 256, 256]], "encoding": "raw", 
        "key": "8", "resolution": [325, 325, 325], "size": [256, 256, 113], "voxel_offset": [0, 0, 0]},
        {"chunk_sizes": [[256, 256, 256]], "encoding": "raw", 
        "key": "7", "resolution": [650, 650, 650], "size": [128, 128, 57], "voxel_offset": [0, 0, 0]},
        {"chunk_sizes": [[256, 256, 256]], "encoding": "raw", 
        "key": "6", "resolution": [1300, 1300, 1300], "size": [64, 64, 29], "voxel_offset": [0, 0, 0]},
        {"chunk_sizes": [[256, 256, 256]], "encoding": "raw", 
        "key": "5", "resolution": [2600, 2600, 2600], "size": [32, 32, 15], "voxel_offset": [0, 0, 0]},
        {"chunk_sizes": [[256, 256, 256]], "encoding": "raw", 
        "key": "4", "resolution": [5200, 5200, 5200], "size": [16, 16, 8], "voxel_offset": [0, 0, 0]},
        {"chunk_sizes": [[256, 256, 256]], "encoding": "raw", 
        "key": "3", "resolution": [10400, 10400, 10400], "size": [8, 8, 4], "voxel_offset": [0, 0, 0]},
        {"chunk_sizes": [[256, 256, 256]], "encoding": "raw", 
        "key": "2", "resolution": [20800, 20800, 20800], "size": [4, 4, 2], "voxel_offset": [0, 0, 0]},
        {"chunk_sizes": [[256, 256, 256]], "encoding": "raw", 
        "key": "1", "resolution": [41600, 41600, 41600], "size": [2, 2, 1], "voxel_offset": [0, 0, 0]},
        {"chunk_sizes": [[256, 256, 256]], "encoding": "raw", 
        "key": "0", "resolution": [83200, 83200, 41600], "size": [1, 1, 1], "voxel_offset": [0, 0, 0]},
    ],
    "type": "image", 
}

with open(os.path.join(output,"info"), 'w') as info_file:
    json.dump(info, info_file)
encodedvolume = ngvolume.generate_recursive_chunked_representation(volume, 
                                                                   info, 
                                                                   dtype=volume.dtype, 
                                                                   directory=output,
                                                                   blurring_method='average')



```
![plot](volume_generation_image.png)
