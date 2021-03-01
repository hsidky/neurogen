import os
import json
import numpy as np

def info_image(directory,
               dtype,
               chunk_size,
               size,
               resolution=[325,325,325]):
    """ 
    This function generates the info file to be used for images 
    in Neuroglancer.
    Parameters
    ----------
    directory : str
        Output directory that info file will be saved to
    dtype : type
        Datatype of the input
    chunk_size : list
        Size of chunks
    size : list
        Size of input data
    resolution : list
        Resolution of input data in nanometers
        If none, defaults to [325, 325, 325] (nm)
    """

    # The number of scales based on the size of the input
    scales = []
    num_scales = np.floor(np.log2(max(size))).astype('int')

    # Specify scale parameters
    for i in range(num_scales+1):
        scale = {}
        scale['chunk_sizes'] = [chunk_size]
        scale['encoding'] = 'raw'
        scale['key'] = str(num_scales-i)
        scale['size'] = [int(np.ceil(k/(2**i))) for k in size[:3]]
        if i == 0:
            scale['resolution'] = [float(resolution[res]) for res in range(3)]
        else:
            previous_scale = scales[-1]
            previous_size = previous_scale["size"]
            previous_resolution = previous_scale["resolution"]
            scale['resolution'] = [(np.ceil(previous_size[res]/scale['size'][res])*previous_resolution[res]).astype('float') 
                                    for res in range(3)]

        scale['voxel_offset'] = [0,0,0]

        scales.append(scale)

    # Header and other details
    info = {
        '@type': 'neuroglancer_multiscale_volume',
        'data_type': str(dtype),
        'num_channels': '1',
        'scales': scales,
        'type': 'image'
    }

    # Write the info file to appropriate directory
    with open(os.path.join(directory,"info"), 'w') as info_file:
        json.dump(info, info_file)
    
    # return dictionary
    return info
    
