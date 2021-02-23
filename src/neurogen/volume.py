import numpy as np
import skimage.measure
import os
from pathlib import Path
import math
import traceback
import logging
import javabridge as jutil
from concurrent.futures import ThreadPoolExecutor

def encode_volume(chunk):
    """ Encode a chunk from a Numpy array into bytes.
    Inputs:
        chunk - array with four dimensions (C, Z, Y, X)
    Outputs:
        buf - encoded chunk (byte stream)
    """
    # Rearrange the image for Neuroglancer
    chunk = np.moveaxis(chunk.reshape(chunk.shape[0],chunk.shape[1],chunk.shape[2],1),
                        (0, 1, 2, 3), (3, 2, 1, 0))
    assert chunk.ndim == 4
    buf = chunk.tobytes()
    return buf

def _mode3(image):
    """ Find mode of pixels in optical field 2x2x2 and stride 2
    This method works by finding the largest number that occurs at least twice
    in a 2x2x2 grid of pixels, then sets that value to the output pixel.
    Inputs:
        image - numpy array with three dimensions (m,n,p)
    Outputs:
        mode_img - numpy array with only two dimensions (round(m/2),round(n/2),round(p/2))
    """
    def forloop(mode_img, idxfalse, vals):
        for i in range(7):
            rvals = vals[i]
            for j in range(i+1,8):
                cvals = vals[j]
                ind = np.logical_and(cvals==rvals,rvals>mode_img[idxfalse])
                mode_img[idxfalse][ind] = rvals[ind]
        return mode_img

    dtype = image.dtype
    imgshape = image.shape
    ypos, xpos, zpos = imgshape

    y_edge = ypos % 2
    x_edge = xpos % 2
    z_edge = zpos % 2

    # Initialize the mode output image (Half the size)
    mode_imgshape = np.ceil([d/2 if d > 1 else 2 for d in imgshape ]).astype('int')
    mode_img = np.zeros(mode_imgshape).astype('uint16')

    # Garnering the eight different pixels that we would find the modes of
    # Finding the mode of: 
    # vals000[1], vals010[1], vals100[1], vals110[1], vals001[1], vals011[1], vals101[1], vals111[1]
    # vals000[2], vals010[2], vals100[2], vals110[2], vals001[2], vals011[2], vals101[2], vals111[2]
    # etc 
    vals000 = image[0:-1:2, 0:-1:2,0:-1:2]
    vals010 = image[0:-1:2, 1::2,0:-1:2]
    vals100 = image[1::2,   0:-1:2,0:-1:2]
    vals110 = image[1::2,   1::2,0:-1:2]
    vals001 = image[0:-1:2, 0:-1:2,1::2]
    vals011 = image[0:-1:2, 1::2,1::2]
    vals101 = image[1::2,   0:-1:2,1::2]
    vals111 = image[1::2,   1::2,1::2]

    # Finding all quadrants where at least two of the pixels are the same
    index = ((vals000 == vals010) & (vals000 == vals100)) & (vals000 == vals110) 
    indexfalse = index==False
    indextrue = index==True

    # Going to loop through the indexes where the two pixels are not the same
    valueslist = [vals000[indexfalse], vals010[indexfalse],
                vals100[indexfalse], vals110[indexfalse],
                vals001[indexfalse], vals011[indexfalse],
                vals101[indexfalse], vals111[indexfalse]]
    edges = (y_edge,x_edge,z_edge)

    mode_edges = {
        (0,0,0): mode_img[:, :, :],
        (0,1,0): mode_img[:,:-1,:],
        (1,0,0): mode_img[:-1,:,:],
        (1,1,0): mode_img[:-1,:-1,:],
        (0,0,1): mode_img[:,:,:-1],
        (0,1,1): mode_img[:,:-1,:-1],
        (1,0,1): mode_img[:-1,:,:-1],
        (1,1,1): mode_img[:-1, :-1, :-1]
    }

    # Edge cases, if there are an odd number of pixels in a row or column, 
    #     then we ignore the last row or column
    # Those columns will be black
    if edges == (0,0,0):
        mode_img[indextrue] = vals000[indextrue]
        mode_img = forloop(mode_img, indexfalse, valueslist)
        return mode_img
    else:
        shortmode_img = mode_edges[edges]
        shortmode_img[indextrue] = vals000[indextrue]
        shortmode_img = forloop(shortmode_img, indexfalse, valueslist)
        mode_edges[edges] = shortmode_img
        return mode_edges[edges]


def _avg3(image):
    """ Average pixels together with optical field 2x2x2 and stride 2
    
    Inputs:
        image - numpy array with only three dimensions (m,n,p)
    Outputs:
        avg_img - numpy array with only three dimensions (round(m/2),round(n/2),round(p/2))
    """

    # Cast to appropriate type for safe averaging
    if image.dtype == np.uint8:
        dtype = np.uint16
    elif image.dtype == np.uint16:
        dtype = np.uint32
    elif image.dtype == np.uint32:
        dtype = np.uint64
    elif image.dtype == np.int8:
        dtype = np.int16
    elif image.dtype == np.int16:
        dtype = np.int32
    elif image.dtype == np.int32:
        dtype = np.int64
    else:
        dtype = image.dtype

    # Store original data type, and cast to safe data type for averaging
    odtype = image.dtype
    image = image.astype(dtype)
    imgshape = image.shape

    # Account for dimensions with odd dimensions to prevent data loss
    ypos = imgshape[0]
    xpos = imgshape[1]
    zpos = imgshape[2]
    z_max = zpos - zpos % 2    # if even then subtracting 0. 
    y_max = ypos - ypos % 2    # if odd then subtracting 1
    x_max = xpos - xpos % 2
    yxz_max = [y_max, x_max, z_max]

    # Initialize the output
    avg_imgshape = np.ceil([d/2 for d in imgshape]).astype(int)
    avg_img = np.zeros(avg_imgshape,dtype=dtype)

    # Do the work
    avg_img[0:int(y_max/2),0:int(x_max/2),0:int(z_max/2)]= (
        image[0:y_max-1:2,0:x_max-1:2,0:z_max-1:2] + 
        image[1:y_max:2  ,0:x_max-1:2,0:z_max-1:2] + 
        image[0:y_max-1:2,1:x_max:2  ,0:z_max-1:2] + 
        image[1:y_max:2  ,1:x_max:2  ,0:z_max-1:2] + 
        image[0:y_max-1:2,0:x_max-1:2,1:z_max:2  ] + 
        image[1:y_max:2  ,0:x_max-1:2,1:z_max:2  ] + 
        image[0:y_max-1:2,1:x_max:2  ,1:z_max:2  ] + 
        image[1:y_max:2  ,1:x_max:2  ,1:z_max:2  ]
    )/8

    # Account for odd shaped dimensions to prevent data loss
    # TODO: This accounts for edge planes, but not edge lines and corners
    if z_max != image.shape[2]:
        avg_img[:int(y_max/2),:int(x_max/2),-1] = (image[0:y_max-1:2,0:x_max-1:2,-1] + 
                                                   image[1:y_max:2  ,0:x_max-1:2,-1] + 
                                                   image[0:y_max-1:2,1:x_max:2  ,-1] + 
                                                   image[1:y_max:2  ,1:x_max:2  ,-1])/4
    if y_max != image.shape[0]:
        avg_img[-1,:int(x_max/2),:int(z_max/2)] = (image[-1,0:x_max-1:2,0:z_max-1:2] + \
                                                   image[-1,0:x_max-1:2,1:z_max:2  ] + \
                                                   image[-1,1:x_max:2  ,0:z_max-1:2] + \
                                                   image[-1,1:x_max:2  ,1:z_max:2  ])/4
    if x_max != image.shape[1]:
        avg_img[:int(y_max/2),-1,:int(z_max/2)] = (image[0:y_max-1:2,-1,0:z_max-1:2] + \
                                                   image[0:y_max-1:2,-1,1:z_max:2  ] + \
                                                   image[1:y_max:2  ,-1,0:z_max-1:2] + \
                                                   image[1:y_max:2  ,-1,1:z_max:2  ])/4
    if (y_max != image.shape[0] and x_max != image.shape[1]) and (z_max != image.shape[2]):
        avg_img[-1,-1,-1] = image[-1,-1,-1]

    return avg_img.astype(odtype)

def generate_chunked_representation(volume,
                                    info,
                                    directory,
                                    blurring_method='mode'):
    """ Generates pyramids of the volume 
    https://en.wikipedia.org/wiki/Pyramid_(image_processing) 

    Parameters
    ----------
    volume : numpy array
        A 3D numpy array representing a volume
    info : dict
        The "info JSON file specification" that is required by Neuroglancer as a dict.
    directory : str
        Neuroglancer precomputed volume directory.
    mode : str
        Either the average or the mode is taken for blurring to generate the pyramids.
        Average - better for Images
        Mode - better for Labelled Data

    Returns
    -------
    Pyramids of the volume with a chunk representation as specified by the info JSON 
    file specification in the output directory.
    """

    # Initialize information from info file
    num_scales = len(info['scales'])

    # Loop through all scales
    # Generate volumes of highest resolution first
    for i in range(num_scales):
        
        # Intialize information from this scale
        key = info['scales'][i]['key']
        size = info['scales'][i]['size']
        chunk_size = info['scales'][i]['chunk_sizes'][0]

        # Number of chunks in every dimension
        num_chunks_x = math.ceil(size[0]/chunk_size[0]) 
        num_chunks_y = math.ceil(size[1]/chunk_size[1])
        num_chunks_z = math.ceil(size[2]/chunk_size[2])
        num_chunks = num_chunks_x * num_chunks_y * num_chunks_z
        volumeshape = volume.shape

        # Chunk Values
        xsplits = np.arange(0, volumeshape[1], chunk_size[0])
        ysplits = np.arange(0, volumeshape[0], chunk_size[1])
        zsplits = np.arange(0, volumeshape[2], chunk_size[2])

        # Initialize directory for scale
        volume_dir = os.path.join(directory, str(key))
        os.makedirs(volume_dir, exist_ok=True)

        # Initalize volume for next scale
        newvolume_shape = np.ceil([d/2 for d in volume.shape]).astype(int)
        newvolume = np.zeros(newvolume_shape,dtype=volume.dtype)

        for x in range(num_chunks_x):
            for y in range(num_chunks_y):
                for z in range(num_chunks_z):
                    start_x, start_y, start_z = xsplits[x], ysplits[y], zsplits[z]
                    # range(num_chunks) does not include ending bounds, must specify
                    try:
                        end_x = xsplits[x+1]
                    except:
                        end_x = volumeshape[0]
                    try:
                        end_y = ysplits[y+1]
                    except:
                        end_y = volumeshape[1]
                    try:
                        end_z = zsplits[z+1]
                    except:
                        end_z = volumeshape[2]
                    
                    # chunk of volume is saved to directory
                    subvolume = volume[start_x:end_x, 
                                       start_y:end_y,
                                       start_z:end_z]

                    encoded_subvolume = encode_volume(np.expand_dims(subvolume, 3))
                    file_name = "{}-{}_{}-{}_{}-{}".format(start_x, end_x, start_y, end_y, start_z, end_z)
                    with open(os.path.join(volume_dir, f'{file_name}'), 'wb') as chunk_storage:
                        chunk_storage.write(encoded_subvolume)

                    # For the next level of detail, the chunk is "blurred" and saved to new volume
                    blurred_volume = None
                    if blurring_method == 'mode':
                        blurred_subvolume = _mode3(subvolume)
                    else:
                        blurred_subvolume = _avg3(subvolume)
                    blurred_shape = blurred_subvolume.shape

                    # Specify bounds for new volume
                    new_start_x = int(start_x/2)
                    new_start_y = int(start_y/2)
                    new_start_z = int(start_z/2)
                    new_end_x = new_start_x + blurred_shape[0]
                    new_end_y = new_start_y + blurred_shape[1]
                    new_end_z = new_start_z + blurred_shape[2] 
                    
                    # Save values to new volume
                    newvolume[new_start_x:new_end_x,
                              new_start_y:new_end_y,
                              new_start_z:new_end_z] = blurred_subvolume

        # update current to be averaged volume (which is half in size)
        volume = newvolume
    return volume
