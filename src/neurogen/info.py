import os
import json
import numpy as np

class class_info:
    def __init__(self, dtype, chunk_size, size, resolution):
        scales = scaling(chunk_size,
                         size,
                         resolution)
        self.info = {
            '@type': 'neuroglancer_multiscale_volume',
            'data_type': str(dtype),
            'num_channels': '1',
            'scales': scales
        }

class image_info(class_info):
     def __init__(self, dtype, chunk_size, size, resolution):
        class_info.__init__(self, dtype, chunk_size, size, resolution)
        info = self.info
        info['type'] =  "image"

class segmentation_info(class_info):

    def __init__(self, dtype, chunk_size, size, resolution):
        class_info.__init__(self, dtype, chunk_size, size, resolution)
        info = self.info
        info['type'] =  "segmentation"
    
    def segment_properties(self, ids, labelled_ids):

        str_ids = [str(id) for id in ids]
        if (labelled_ids == None):
            labelled_ids = ["ID_"+str(id) for id in ids]

        ids_info = {
        "ids": str_ids,
        "properties":[
            {
            "id":"label",
            "type":"label",
            "values": labelled_ids
            },
            {
            "id":"description",
            "type":"label",
            "values": str_ids
            }
            ]
        }

        self.segmentation_info = {
            "@type": "neuroglancer_segment_properties",
            "inline": ids_info
        }


def scaling(chunk_size,
            size,
            resolution):
    """
    This function generates the necessary information for 
    scaling the input data.
    
    Parameters
    ----------
    chunk_size : list
        Size of chunks
    size : list
        Size of input data
    resolution : list
        Resolution of input data in nanometers
        If none, defaults to [325, 325, 325] (nm)

    Returns
    -------
    scales : list
        A list of all the scales concatenated in order
    """
    # The number of scales based on the size of the input
    num_scales = np.floor(np.log2(max(size))).astype('int')

    scales = []
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
    
    return scales


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

    info_class = image_info(dtype, chunk_size, size, resolution)
    info = info_class.info
    

    # Write the info file to appropriate directory
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory,"info"), 'w') as info_file:
        json.dump(info, info_file)

    # return dictionary
    return info


def info_segmentation(ids,
                      directory,
                      dtype,
                      chunk_size,
                      size,
                      segmentation_directory,
                      resolution=[325,325,325],
                      labelled_ids=None):
    """ 
    This function generates the info file to be used for images 
    in Neuroglancer.
    Parameters
    ----------
    ids : list
        List of the labelled ids in input dataset
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
    labelled_ids : list
        List of the names to label each segment id
    """

    info_class = segmentation_info(dtype, chunk_size, size, resolution)

    info = info_class.info
    segmentation_info = segmentation_info.segment_properties 

    # Write the info file to appropriate directory
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory,"info"), 'w') as info_file:
        json.dump(info, info_file)

    # Write the segment properties to appropriate sub-directory
    output = os.path.join(directory,segmentation_directory)
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)
    with open(os.path.join(output,"info"), "w") as segment_info_file:
        json.dump(segmentation_info, segment_info_file)
    
    # return dictionary
    return info