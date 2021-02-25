import numpy as np
import skimage.measure
import os
from pathlib import Path
from javabridge import jutil
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


def write_image(image, volume_directory, scale, x, y, z):
    if not os.path.exists(volume_directory):
            os.makedirs(volume_directory, exist_ok=True)
    file_name = "{}-{}_{}-{}_{}-{}".format(str(x[0]), str(x[1]),
                                           str(y[0]), str(y[1]),
                                           str(z[0]), str(z[1]))
    with open(os.path.join(volume_directory, f'{file_name}'), 'wb') as chunk_storage:
                    chunk_storage.write(image)


def generate_iterative_chunked_representation(volume,
                                    info,
                                    directory,
                                    blurring_method='mode',
                                    blurred_image=None):

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
    blurring_method : str
        Either the average or the mode is taken for blurring to generate the pyramids.
        Average - better for Images
        Mode - better for Labelled Data

    Returns
    -------
    Pyramids of the volume with a chunk representation as specified by the info JSON 
    file specification in the output directory.
    """

    size_0 = (info['scales'][0]['size'])
    volumeshape_0 = list(volume.shape[:3])
    if (volumeshape_0 != size_0):
        raise ValueError("Make sure the (X,Y,Z) axis for volume shape {} matches the info file specification size {}".format(volumeshape_0, size_0))

    if blurred_image == None:
        blurred_image = np.zeros(volume.shape, dtype=volume.dtype)

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
        volumeshape = volume.shape

        # Chunk Values
        xsplits = list(np.arange(0, volumeshape[0], chunk_size[0]))
        xsplits.append(volumeshape[0])
        ysplits = list(np.arange(0, volumeshape[1], chunk_size[1]))
        ysplits.append(volumeshape[1])
        zsplits = list(np.arange(0, volumeshape[2], chunk_size[2]))
        zsplits.append(volumeshape[2])

        # Initialize directory for scale
        volume_dir = os.path.join(directory, str(key))

        blurred_image_shape = [int(np.ceil(d/2)) for d in volume.shape]
        
        for y in range(len(ysplits)-1):
            for x in range(len(xsplits)-1):
                for z in range(len(zsplits)-1):
                    start_x, end_x = (xsplits[x], xsplits[x+1])
                    start_y, end_y = (ysplits[y], ysplits[y+1])
                    start_z, end_z = (zsplits[z], zsplits[z+1])

                    # chunk of volume is saved to directory
                    subvolume = volume[start_x:end_x, 
                                       start_y:end_y,
                                       start_z:end_z]
                    subvolume = subvolume.reshape(subvolume.shape[:3])
                    
                    encoded_subvolume = encode_volume(np.expand_dims(subvolume, 3))
                    write_image(image=encoded_subvolume, volume_directory=volume_dir,
                                scale=i, x=[start_x, end_x], y=[start_y, end_y], z= [start_z, end_z])

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
                    blurred_image[new_start_x:new_end_x,
                                  new_start_y:new_end_y,
                                  new_start_z:new_end_z,0,0] = blurred_subvolume
                    

        # update current to be averaged volume (which is half in size)
        volume = blurred_image[:blurred_image_shape[0],
                               :blurred_image_shape[1],
                               :blurred_image_shape[2],0,0]

    return volume


def generate_recursive_chunked_representation(volume, info, dtype, directory, blurring_method='mode', S=0, X=None,Y=None,Z=None):

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
        A 3D numpy array representing a volume
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
    if X == None: X = [0,size[0]]
    if Y == None: Y = [0,size[1]]
    if Z == None: Z = [0,size[2]]
    
    # Modify upper bound to stay within resolution dimensions
    if X[1] > size[0]: X[1] = size[0]
    if Y[1] > size[1]: Y[1] = size[1]
    if Z[1] > size[2]: Z[1] = size[2]
    
    # If requesting from the lowest scale, then just read the image
    if str(S)==all_scales[0]['key']:

        # Taking a chunk of the input
        image = volume[X[0]:X[1],Y[0]:Y[1],Z[0]:Z[1]]

        # Encode the chunk
        image_encoded = encode_volume(image)

        # Write the chunk
        volume_dir = os.path.join(directory, str(S))
        write_image(image=image_encoded, volume_directory=volume_dir, 
                    scale=S, x=X, y=Y, z=Z)
        
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
             jutil.attach()
             sub_image = generate_recursive_chunked_representation(**kwargs)
             jutil.detach()
             image = args[0]
             x_ind = args[1]
             y_ind = args[2]
             z_ind = args[3]
             if blurring_method == 'mode':
                image[x_ind[0]:x_ind[1],y_ind[0]:y_ind[1],z_ind[0]:z_ind[1]] = _mode3(sub_image)
             else:
                image[x_ind[0]:x_ind[1],y_ind[0]:y_ind[1],z_ind[0]:z_ind[1]] = _avg3(sub_image)

        with ThreadPoolExecutor(max_workers=8) as executor:
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
                    scale=S, x=X, y=Y, z=Z)

        return image