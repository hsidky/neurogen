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


Download example data and unzip in appropriate input directory:
http://graphics.stanford.edu/data/voldata/cthead-8bit.tar.gz

More details of data: http://graphics.stanford.edu/data/voldata/

```python
import numpy as np
import os
from PIL import Image
import volume as ngvolume
import info as nginfo


# Unzip tar  file into appropriate directory
input = "/home/ubuntu/neurogen/src/neurogen/cthead_input/"
output = "/home/ubuntu/neurogen/src/neurogen/output/"

#  Generate Input
volume = np.zeros((256,256,113,1,1)).astype('uint8')
for filename in os.listdir(input):
    index = int(filename[11:14])
    filename = os.path.join(input,filename)
    image = Image.open(filename)
    imarray = np.array(image)
    volume[:,:,index,0,0] = imarray.astype('uint8')

volume_info = nginfo.info_image(directory=output, 
                                dtype=volume.dtype, 
                                chunk_size=[256,256,256],
                                size=volume.shape)

encodedvolume = ngvolume.generate_recursive_chunked_representation(volume, 
                volume_info, dtype=volume.dtype, directory=output,blurring_method='average')





```
![plot](volume_generation_image.png)
