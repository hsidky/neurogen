import os
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import itertools
from itertools import repeat

import traceback

import bfio
from bfio import BioReader, BioWriter


def encode_volume(chunk):
    """ Encode a chunk from a Numpy array into bytes.
    Parameters 
    ----------
    chunk : numpy array
        Array with three dimensions (Z, Y, X)
    Returns
    -------
    buf : byte stream
        The encoded chunk compatible with Neuroglancer
    """
    # Rearrange the image for Neuroglancer
    chunk = np.moveaxis(chunk.reshape(chunk.shape[0],chunk.shape[1],chunk.shape[2],1),
                        (0, 1, 2, 3), (3, 2, 1, 0))
    assert chunk.ndim == 4
    buf = chunk.tobytes()
    return buf


def decode_volume(encoded_volume, dtype, shape=None):
    """ Decodes an enocoded image for Neuroglancer
    Parameters
    ----------
    encoded_volume : byte stream
        The encoded volume that needs to be decoded .
    dtype : type
        The datatype of the byte stream (encoded_volume).
    shape : numpy array
        3D shape that byte stream gets rearranged to.
    Returns 
    -------
    decoded_volume : numpy array
        The decoded input 
    """

    decoded_volume = np.frombuffer(encoded_volume, dtype=dtype)
    decoded_volume = decoded_volume.reshape((shape[2], shape[1], shape[0]))
    decoded_volume = np.moveaxis(decoded_volume, (0,1,2), (2,1,0))

    return decoded_volume


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
    mode_imgshape = np.ceil([d/2 for d in imgshape]).astype('int')
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
        mode_edge_shape = mode_edges[edges].shape
        if (mode_imgshape != mode_edge_shape).all():
            mode_img[:mode_edge_shape[0],
                    :mode_edge_shape[1],
                    :mode_edge_shape[2]] = mode_edges[edges]
        return mode_img


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


def write_image(image, volume_directory, x, y, z):
    """ This function writes the images to the volume directories 
    
    Parameters
    ----------
    image - numpy array
        Array that is being converted and written
    volume - str
        Output directory the image is saved to
    x - list
        Range of the x dimension for the image
    y - list 
        Range of the y dimension for the image
    z - list 
        Range of the z dimension for the image
    """

    if not os.path.exists(volume_directory):
            os.makedirs(volume_directory, exist_ok=True)
    file_name = "{}-{}_{}-{}_{}-{}".format(str(x[0]), str(x[1]),
                                           str(y[0]), str(y[1]),
                                           str(z[0]), str(z[1]))
    with open(os.path.join(volume_directory, f'{file_name}'), 'wb') as chunk_storage:
                    chunk_storage.write(image)


def get_rest_of_the_pyramid(directory, 
                            input_shape, 
                            chunk_size, 
                            datatype,
                            blurring_method = 'mode'):
    """ This function builds the next level of the pyramid based on the input.
        Usually this function is used inside a while loop. 
        
            Based on the input directory, this function creates a dictionary to group the 
            that will be blurred together.  Once the images are organized, then it will 
            concatenate all images into one, apply the blurring method, encode the blurred image,
            and save in the next, higher level directory.

    Parameters
    ----------
    directory : str
        The directory of the images that are being read.
    input_shape : numpy array
        The shape of all the chunks combined in 'directory'
    chunk_size : numpy array
        The size of chunks in directory
    datatype : type
        The datatype of the volumes in 'directory'
    blurring_method : str
        Specifies whether we are averaging the images or
        taking the mode.
    
    Returns
    -------
    Saves images into the next level of the pyramid.
    """


    chunk0, chunk1, chunk2 = (chunk_size[0], chunk_size[1], chunk_size[2])
    doublechunk0, doublechunk1, doublechunk2 = (chunk0*2, chunk1*2, chunk2*2)
    in_shape0, in_shape1, in_shape2 = (input_shape[0], input_shape[1], input_shape[2])

    # Initialize the keys of the dictionary that help organize the images in directory
    group_images = {}
    range_x = list(np.arange(0,input_shape[0],doublechunk0))
    range_y = list(np.arange(0,input_shape[1],doublechunk1))
    range_z = list(np.arange(0,input_shape[2],doublechunk2))
    range_x = [(x, x+doublechunk0) if (x+doublechunk0) < in_shape0 else (x, in_shape0) for x in range_x]
    range_y = [(y, y+doublechunk1) if (y+doublechunk1) < in_shape1 else (y, in_shape1) for y in range_y]
    range_z = [(z, z+doublechunk2) if (z+doublechunk2) < in_shape2 else (z, in_shape2) for z in range_z]
    combo_ranges = [range_x, range_y, range_z]
    all_combos = list(itertools.product(*combo_ranges))
    for all_c in all_combos:
        group_images[all_c] = []

    # Sort the images in directory into its repestive dictionary key
    for image in sorted(os.listdir(directory)):
        xyz = image.split("_")
        split_x = xyz[0].split("-")
        split_y = xyz[1].split("-")
        split_z = xyz[2].split("-")

        x0 = int(np.floor(int(split_x[0])/doublechunk0)*doublechunk0)
        x1 = int(np.ceil(int(split_x[1])/doublechunk0)*doublechunk0)
        y0 = int(np.floor(int(split_y[0])/doublechunk1)*doublechunk1)
        y1 = int(np.ceil(int(split_y[1])/doublechunk1)*doublechunk1)
        z0 = int(np.floor(int(split_z[0])/doublechunk2)*doublechunk2)
        z1 = int(np.ceil(int(split_z[1])/doublechunk2)*doublechunk2)

        # Edges need to be fixed of the np.arange function
        if x1 == 0:
            x1 = doublechunk0
        else:
            x1 = min(x1, in_shape0)
        
        if y1 == 0:
            y1 = doublechunk1
        else:
            y1 = min(y1, in_shape1)
        
        if z1 == 0:
            z1 = doublechunk2
        else:
            z1 = min(z1, in_shape2)
        
        # Add the images to the appropriate key
        key = ((x0, x1), (y0, y1), (z0, z1))
        group_images[key].append(image)

    # for every key in the dictionary, blur all the images together and create one output
    base_directory = os.path.basename(directory)
    parent_directory = os.path.dirname(directory)
    new_directory = os.path.join(parent_directory, str(int(base_directory)-1))

    def blur_images_indict(i, dictionary_grouped_imgs, datatype):
        """ This function blurs four images together to give the next level of the pyramid. 
            This function was specifically built so that images that needed to be averaged 
                together could be done with multiprocessing. """

        # Every key needs to be initialized 
        new_image = np.zeros(((i[0][1]-i[0][0]), (i[1][1]-i[1][0]), (i[2][1]-i[2][0]))).astype(datatype)
        index_offset = min(dictionary_grouped_imgs[i]).split("_")
        index_offset = [int(ind.split("-")[0]) for ind in index_offset]
        # iterate through the images that need to be grouped together
        for image in dictionary_grouped_imgs[i]:
            img_edge = image.split("_")
            img_edge = [list(map(int, im.split("-"))) for im in img_edge]
            imgshape = (img_edge[0][1]-img_edge[0][0], img_edge[1][1]-img_edge[1][0], img_edge[2][1]-img_edge[2][0])
            with open(os.path.join(directory, image), "rb") as im:
                decoded_image = decode_volume(encoded_volume=im.read(), dtype=datatype, shape=imgshape)
                new_image[img_edge[0][0]-index_offset[0]:img_edge[0][1]-index_offset[0], 
                          img_edge[1][0]-index_offset[1]:img_edge[1][1]-index_offset[1], 
                          img_edge[2][0]-index_offset[2]:img_edge[2][1]-index_offset[2]] = decoded_image

        # Output the blurred image
        if blurring_method == 'mode':  
            blurred_image = _mode3(new_image).astype(datatype)
        else:
            blurred_image = _avg3(new_image).astype(datatype)

        encoded_image = encode_volume(chunk=blurred_image)
        x_write = list(np.ceil([i[0][0]/2, i[0][1]/2]).astype(np.int))
        y_write = list(np.ceil([i[1][0]/2, i[1][1]/2]).astype(np.int))
        z_write = list(np.ceil([i[2][0]/2, i[2][1]/2]).astype(np.int))

        write_image(image=encoded_image, volume_directory=new_directory, 
                    x=x_write, y=y_write, z=z_write)

    # Go through the dictionary to blur the images list in the dictionary values together
        # use multiprocessing to use multiple cores
    with ThreadPoolExecutor(max_workers = os.cpu_count()-1) as executor:
        executor.map(blur_images_indict, 
                    (i for i in group_images.keys()), 
                    repeat(group_images), 
                    repeat(datatype))
    

def generate_recursive_chunked_representation(
    volume, 
    info, 
    dtype, 
    directory, 
    blurring_method='mode', 
    S=0, 
    X=None,
    Y=None,
    Z=None,
    max_workers=8):

    """ Recursive function for pyramid building
    This is a recursive function that builds an image pyramid by indicating
    an original region of an image at a given scale. This function then
    builds a pyramid up from the highest resolution components of the pyramid
    (the original images) to the given position resolution.
    As an example, imagine the following possible pyramid:
    Scale S=0                     1234
                                 /    \
    Scale S=1                  12      34
                              /  \    /  \
    Scale S=2                1    2  3    4
    At scale 2 (the highest resolution) there are 4 original images. At scale 1,
    images are averaged and concatenated into one image (i.e. image 12). Calling
    this function using S=0 will attempt to generate 1234 by calling this
    function again to get image 12, which will then call this function again to
    get image 1 and then image 2. Note that this function actually builds images
    in quadrants (top left and right, bottom left and right) rather than two
    sections as displayed above.
    
    Due to the nature of how this function works, it is possible to build a
    pyramid in parallel, since building the subpyramid under image 12 can be run
    independently of the building of subpyramid under 34.
    
    Parameters
    ----------
    volume : numpy array
        A 5D numpy array representing a volume.  Order of dimensions depends on info file.
    info : dict
        The "info JSON file specification" that is required by Neuroglancer as a dict.
    dtype : datatype
        The datatype of the input volume 
    directory : str
        Neuroglancer precomputed volume directory.
    blurring_method : str
        Either the average or the mode is taken for blurring to generate the pyramids.
        Average - better for Images
        Mode - better for Labelled Data
    S : int 
        Current scale as specified in info file specification
    X : list
        The range of X indexes of the input volume that is being written.
    Y : list
        The range of Y indexes of the input volume that is being written.
    Z : list 
        The range of Z indexes of the input volume that is being written.
    max_workers: int
        Maximum number of workers to use to construct the pyramid.

    Returns
    -------
    Pyramids of the volume with a chunk representation as specified by the info JSON 
    file specification in the output directory.
    """
    # Scales
    all_scales = info['scales']
    num_scales = len(all_scales)

    # Current scale info
    scale_info = all_scales[num_scales-S-1]
    size = scale_info['size']
    chunk_size = scale_info['chunk_sizes'][0]


    if scale_info==None:
        ValueError("No scale information for resolution {}.".format(S))

    # Initialize X, Y, Z
    if X == None: 
        X = [0,size[0]]
    if Y == None: 
        Y = [0,size[1]]
    if Z == None: 
        Z = [0,size[2]]
    
    # Modify upper bound to stay within resolution dimensions
    if X[1] > size[0]: 
        X[1] = size[0]
    if Y[1] > size[1]: 
        Y[1] = size[1]
    if Z[1] > size[2]: 
        Z[1] = size[2]
    
    # If requesting from the lowest scale, then just read the image
    if str(S)==all_scales[0]['key']:

        # if the volume is a bfio object, then cache
        if type(volume) is bfio.bfio.BioReader:
            if hasattr(volume,'cache') and \
                X[0] >= volume.cache_X[0] and X[1] <= volume.cache_X[1] and \
                Y[0] >= volume.cache_Y[0] and Y[1] <= volume.cache_Y[1] and \
                Z[0] >= volume.cache_Z[0] and Z[1] <= volume.cache_Z[1]:

                pass

            else:
                X_min = 1024 * (X[0]//volume._TILE_SIZE)
                Y_min = 1024 * (Y[0]//volume._TILE_SIZE)
                Z_min = 1024 * (Z[0]//volume._TILE_SIZE)
                X_max = min([X_min+1024,volume.X])
                Y_max = min([Y_min+1024,volume.Y])
                Z_max = min([Z_min+1024,volume.Z])
                
                volume.cache = volume[Y_min:Y_max,X_min:X_max,Z_min:Z_max,0,0].squeeze()
                
                volume.cache_X = [X_min,X_max]
                volume.cache_Y = [Y_min,Y_max]
                volume.cache_Z = [Z_min,Z_max]
            
            # Taking a chunk of the input
            image = volume.cache[Y[0]-volume.cache_Y[0]:Y[1]-volume.cache_Y[0],
                                 X[0]-volume.cache_X[0]:X[1]-volume.cache_X[0],                                  
                                 Z[0]-volume.cache_Z[0]:Z[1]-volume.cache_Z[0]]
        
        else:
            # Taking a chunk of the input
            image = volume[X[0]:X[1],Y[0]:Y[1],Z[0]:Z[1]]

        # Encode the chunk
        image_encoded = encode_volume(image)

        # Write the chunk
        volume_dir = os.path.join(directory, str(S))
        write_image(image=image_encoded, volume_directory=volume_dir, 
                    x=X, y=Y, z=Z)
        
        return image

    else:

        #initialize the output image
        image = np.zeros((X[1]-X[0],Y[1]-Y[0],Z[1]-Z[0],1,1),dtype=dtype)
        
        # Set the subgrid dimensions
        subgrid_x = list(np.arange(2*X[0],2*X[1],chunk_size[0])) 
        subgrid_x.append(2*X[1])
        subgrid_y = list(np.arange(2*Y[0],2*Y[1],chunk_size[1]))
        subgrid_y.append(2*Y[1])
        subgrid_z = list(np.arange(2*Z[0],2*Z[1],chunk_size[2]))
        subgrid_z.append(2*Z[1])

        def load_and_scale(*args,**kwargs):
            sub_image = generate_recursive_chunked_representation(**kwargs)
            image = args[0] 
            x_ind = args[1]
            y_ind = args[2]
            z_ind = args[3]
            if blurring_method == 'mode':
                image[x_ind[0]:x_ind[1],y_ind[0]:y_ind[1],z_ind[0]:z_ind[1]] = _mode3(sub_image)
            else:
                image[x_ind[0]:x_ind[1],y_ind[0]:y_ind[1],z_ind[0]:z_ind[1]] = _avg3(sub_image)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Where each chunk gets mapped to.
            for x in range(0,len(subgrid_x)-1):
                x_ind = np.ceil([(subgrid_x[x]-subgrid_x[0])/2,
                                (subgrid_x[x+1]-subgrid_x[0])/2]).astype('int')
                for y in range(0,len(subgrid_y)-1):
                    y_ind = np.ceil([(subgrid_y[y]-subgrid_y[0])/2,
                                    (subgrid_y[y+1]-subgrid_y[0])/2]).astype('int')
                    for z in range(0,len(subgrid_z)-1):
                        z_ind = np.ceil([(subgrid_z[z]-subgrid_z[0])/2,
                                        (subgrid_z[z+1] - subgrid_z[0])/2]).astype('int')
                        executor.submit(load_and_scale, 
                                        image, x_ind, y_ind, z_ind, 
                                        X=subgrid_x[x:x+2],
                                        Y=subgrid_y[y:y+2],
                                        Z=subgrid_z[z:z+2],
                                        S=S+1,
                                        blurring_method=blurring_method,
                                        directory=directory,
                                        dtype=dtype,
                                        info=info,
                                        volume=volume)

        # Encode the chunk
        image_encoded = encode_volume(image)
        volume_dir = os.path.join(directory, str(S))
        write_image(image=image_encoded, volume_directory=volume_dir,
                    x=X, y=Y, z=Z)

        return image
