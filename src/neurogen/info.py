import os
import json
import numpy as np

class class_info:
    """ This class initializes the header for the info json specification file
    https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md """

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
    """ This class inherits class_info and add the type """ 

    def __init__(self, dtype, chunk_size, size, resolution):
        class_info.__init__(self, dtype, chunk_size, size, resolution)
        info = self.info
        info['type'] =  "image"

class segmentation_info(class_info):
    """ This class inherits class_info and add the type
        It also creates labels for the segments if specified """ 

    def __init__(self, dtype, chunk_size, size, resolution):
        class_info.__init__(self, dtype, chunk_size, size, resolution)
        info = self.info
        info['type'] =  'segmentation'
        self.info = info

    def get_segment_properties(self, ids, labelled_ids, subdirectory):

        info = self.info
        info['segment_properties'] = subdirectory
        self.info = info

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
        return self.info, self.segmentation_info

class mesh_info(class_info):
    """ This class inherits class_info and add the type for segmentation.
        It also creates labels for the segments if specified """ 

    def __init__(self, dtype, chunk_size, size, resolution, mesh_subdirectory):

        class_info.__init__(self, dtype, chunk_size, size, resolution)
        info = self.info
        info['mesh'] = mesh_subdirectory
        info['type'] = 'segmentation'
        self.info = info
        
    def get_multires_mesh_format(self, bit_depth):
        """ More information can be found on:
        https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md#multi-resolution-mesh-format """
        
        info = self.info
        resolution = info['scales'][0]['resolution']

        self.multires_mesh_format = {
            "@type" : "neuroglancer_multilod_draco",
            "lod_scale_multiplier" : 1,
            "transform" : [resolution[0], 0, 0,0,
                           0, resolution[1], 0,0,
                           0, 0, resolution[2],0],
            "vertex_quantization_bits" : bit_depth 
            }
        
        return self.multires_mesh_format

    def get_segment_properties(self, ids, labelled_ids, segmentation_subdirectory):
        info = self.info
        info['segment_properties'] = segmentation_subdirectory
        self.info = info

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
        
        return self.info, self.segmentation_info


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

    image_json = image_info(dtype, chunk_size, size, resolution)
    info = image_json.info
    

    # Write the info file to appropriate directory
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory,"info"), 'w') as info_file:
        json.dump(info, info_file)

    # return dictionary
    return info


def info_segmentation(directory,
                      dtype,
                      chunk_size,
                      size,
                      resolution=[325,325,325],
                      ids=None,
                      labelled_ids=None,
                      segmentation_subdirectory=None):
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
    ids : list
        List of the labelled ids in input dataset
    labelled_ids : list
        List of the names to label each segment id
    segment_subdirectory : str
        Subdirectory of info file for segment_properties
    """

    if (segmentation_subdirectory) or (ids) or (labelled_ids):
        assert segmentation_subdirectory
        assert ids.any()
        # assert labelled_ids.any()

    segmentation_json = segmentation_info(dtype, chunk_size, size, resolution)
    
    # segment properties is optional
    if segmentation_subdirectory == None:
        info = segmentation_json.info
    else:
        info, segment_properties = segmentation_json.get_segment_properties(ids,labelled_ids,segmentation_subdirectory)

    # Write the info file to appropriate directory
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory,"info"), 'w') as info_file:
        json.dump(info, info_file)

    if (segmentation_subdirectory != None):
        # Write the segment properties to appropriate sub-directory
        output = os.path.join(directory,segmentation_subdirectory)
        if not os.path.exists(output):
            os.makedirs(output, exist_ok=True)
        with open(os.path.join(output,"info"), "w") as segment_info_file:
            json.dump(segment_properties, segment_info_file)
    
    # return dictionary
    return info

def info_mesh(directory,
              dtype,
              chunk_size,
              size,
              resolution = [325,325,325],
              ids=None,
              labelled_ids = None,
              mesh_subdirectory='meshdir',
              bit_depth=16,
              segmentation_subdirectory=None):

    """ 
    This function generates the info files and initializes the directories 
    for meshes in Neuroglancer.

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
    ids : list
        List of the labelled ids in input dataset
    labelled_ids : list
        List of the names to label each segment id
    mesh_subdirectory : str
        Subdirectory of meshes
    bit_depth : int
        Number of bits for mesh vertex quantization. Can only be 10 or 16. 
    segment_subdirectory : str
        Subdirectory of info file for segment_properties
    """

    if (segmentation_subdirectory) or (ids) or (labelled_ids):
        assert segmentation_subdirectory
        assert ids.any()
    
    mesh_json = mesh_info(dtype=dtype, 
                          chunk_size=chunk_size, 
                          size=size, 
                          resolution=resolution, 
                          mesh_subdirectory=mesh_subdirectory)

    if segmentation_subdirectory == None:
        info = mesh_json.info
        multires_mesh_format = mesh_json.get_multires_mesh_format(bit_depth=bit_depth)
    else:
        info, segment_properties = mesh_json.get_segment_properties(ids, labelled_ids, segmentation_subdirectory)
        multires_mesh_format = mesh_json.get_multires_mesh_format(bit_depth=bit_depth)


    # Write the info file to appropriate directory
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory,"info"), 'w') as info_file:
        json.dump(info, info_file)

    output_mesh  = os.path.join(directory, mesh_subdirectory)
    if not os.path.exists(output_mesh):
        os.makedirs(output_mesh, exist_ok=True)
    with open(os.path.join(output_mesh, "info"), 'w') as mesh_info_file:
        json.dump(multires_mesh_format,mesh_info_file)
    
    if (segmentation_subdirectory != None):
        # Write the segment properties to appropriate sub-directory
        output = os.path.join(directory,segmentation_subdirectory)
        if not os.path.exists(output):
            os.makedirs(output, exist_ok=True)
        with open(os.path.join(output,"info"), "w") as segment_info_file:
            json.dump(segment_properties, segment_info_file)
            
    return info
