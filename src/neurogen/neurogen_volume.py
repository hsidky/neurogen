import numpy as np
import skimage.measure
import os
from pathlib import Path
import math
import traceback
import logging
import javabridge as jutil
from concurrent.futures import ThreadPoolExecutor

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

CHUNK_SIZE = 256

def encode(chunk):
    """ Encode a chunk from a Numpy array into bytes.
    Inputs:
        chunk - array with four dimensions (C, Z, Y, X)
    Outputs:
        buf - encoded chunk (byte stream)
    """
    # Rearrange the image for Neuroglancer
    chunk = np.moveaxis(chunk.reshape(chunk.shape[0],chunk.shape[1],chunk.shape[2],1),
                        (0, 1, 2, 3), (2, 3, 1, 0))
    # chunk = np.asarray(chunk).astype(chunk.dtype, casting="safe")
    assert chunk.ndim == 4
    # assert chunk.shape[0] == self.num_channels
    buf = chunk.tobytes()
    return buf

class PyramidWriter():
    """ Pyramid file writing base class
    This class should not be called directly. It should be inherited by a pyramid
    writing class type.
    
    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    can_write = True
    chunk_pattern = None

    def __init__(self, base_dir):
        self.base_path = Path(base_dir)

    def store_chunk(self, buf, key, chunk_coords):
        """ Store a pyramid chunk
        
        Inputs:
            buf - byte stream to save to disk
            key - pyramid scale, folder to save chunk to
            chunk_coords - X,Y,Z coordinates of data in buf
        """
        try:
            self._write_chunk(key,chunk_coords,buf)
        except OSError as exc:
            raise FileNotFoundError(
                "Error storing chunk {0} in {1}: {2}" .format(
                    self._chunk_path(key, chunk_coords),
                    self.base_path, exc))

    def _chunk_path(self, key, chunk_coords, pattern=None):
        if pattern is None:
            pattern = self.chunk_pattern
        chunk_coords = self._chunk_coords(chunk_coords)
        chunk_filename = pattern.format(*chunk_coords, key=key)
        return self.base_path / chunk_filename

    def _chunk_coords(self,chunk_coords):
        return chunk_coords

    def _write_chunk(self,key,chunk_path,buf):
        NotImplementedError("_write_chunk was never implemented.")

class NeuroglancerWriter(PyramidWriter):
    """ Method to write a Neuroglancer pre-computed pyramid
    
    Inputs:
        base_dir - Where pyramid folders and info file will be stored
    """

    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.chunk_pattern = "{key}/{0}-{1}_{2}-{3}_{4}-{5}"

    def _write_chunk(self,key,chunk_coords,buf):
        chunk_path = self._chunk_path(key,chunk_coords)
        os.makedirs(str(chunk_path.parent), exist_ok=True)
        with open(str(chunk_path.with_name(chunk_path.name)),'wb') as f:
            # buf = buf.copy(order='C')
            f.write(buf)

def encode_volume(volume):
    assert volume.ndim == 4
    buf = volume.tobytes()
    return buf

def _mode3(image, dtype):
    """ Find mode of pixels in optical field 2x2 and stride 2
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
        # then we ignore the last row or column
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
        image - numpy array with only two dimensions (m,n)
    Outputs:
        avg_img - numpy array with only two dimensions (round(m/2),round(n/2))
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
    avg_imgshape = np.floor([d/2 for d in imgshape]).astype(int)
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
                                    directory):
    
    num_scales = len(info['scales'])
    for i in range(num_scales):
        key = info['scales'][i]['key']
        size = info['scales'][i]['size']
        chunk_size = info['scales'][i]['chunk_sizes'][0]
        num_chunks_x = math.ceil(size[0]/chunk_size[0]) 
        num_chunks_y = math.ceil(size[1]/chunk_size[1])
        num_chunks_z = math.ceil(size[2]/chunk_size[2])
        num_chunks = num_chunks_x * num_chunks_y * num_chunks_z
        volumeshape = volume.shape

        print("KEY {}".format(key))

        xsplits = np.arange(0, volumeshape[1], chunk_size[0])
        ysplits = np.arange(0, volumeshape[0], chunk_size[1])
        zsplits = np.arange(0, volumeshape[2], chunk_size[2])

        volume_dir = os.path.join(directory, str(key))
        os.makedirs(volume_dir, exist_ok=True)

        newvolume_shape = np.ceil([d/2 for d in volume.shape]).astype(int)
        newvolume = np.zeros(newvolume_shape,dtype=volume.dtype)

        for x in range(num_chunks_x):
            for y in range(num_chunks_y):
                for z in range(num_chunks_z):
                    start_x, start_y, start_z = xsplits[x], ysplits[y], zsplits[z]
                    try:
                        end_x = xsplits[x+1]
                    except:
                        end_x = volumeshape[1]
                    try:
                        end_y = ysplits[y+1]
                    except:
                        end_y = volumeshape[0]
                    try:
                        end_z = zsplits[z+1]
                    except:
                        end_z = volumeshape[2]
                    
                    subvolume = volume[start_x:end_x, 
                                       start_y:end_y,
                                       start_z:end_z]
                    

                    average_subvolume = _mode3(subvolume, subvolume.dtype)
                    average_shape = average_subvolume.shape

                    encoded_subvolume = encode(np.expand_dims(subvolume, 3))
                    file_name = "{}-{}_{}-{}_{}-{}".format(start_x, end_x, start_y, end_y, start_z, end_z)
                    with open(os.path.join(volume_dir, f'{file_name}'), 'wb') as chunk_storage:
                        chunk_storage.write(encoded_subvolume)

                    new_start_x = int(start_x/2)
                    new_start_y = int(start_y/2)
                    new_start_z = int(start_z/2)
                    end_start_x = new_start_x + average_shape[0]
                    end_start_y = new_start_y + average_shape[1]
                    end_start_z = new_start_z + average_shape[2]
                    print("VOLUME SHAPE {} - NEXT LEVEL SHAPE {}".format(volume.shape, newvolume.shape))
                    print("SUBVOLUME ", subvolume.shape)
                    print("AVERAGED SUBVOLUME ", average_subvolume.shape)
                    newvolume[new_start_x:end_start_x,
                              new_start_y:end_start_y,
                              new_start_z:end_start_z] = average_subvolume
                    print("X {}-{} is now {}-{}".format(start_x, end_x, new_start_x, end_start_x))
                    print("Y {}-{} is now {}-{}".format(start_y, end_y, new_start_y, end_start_y))
                    print("Z {}-{} is now {}-{}".format(start_z, end_z, new_start_z, end_start_z))
                    print(" ")
                    
        volume = newvolume
        # print(volume.shape)
        print("")
    return volume

def _get_higher_res(vol, slide_writer,encoder, dtype, S=0, X=None,Y=None,Z=None):
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
    
    Inputs:
        S - Top level scale from which the pyramid will be built
        bfio_reader - BioReader object used to read the tiled tiff
        slide_writer - SlideWriter object used to write pyramid tiles
        encoder - ChunkEncoder object used to encode numpy data to byte stream
        X - Range of X values [min,max] to get at the indicated scale
        Y - Range of Y values [min,max] to get at the indicated scale
    Outputs:
        image - The image corresponding to the X,Y values at scale S
    """

    voxel_size = np.float32(encoder['scales'][0]['resolution'])

    scale_info = None
    for res in encoder['scales']:
        if int(res['key'])==S:
            scale_info = res
            break
    if scale_info==None:
        ValueError("No scale information for resolution {}.".format(S))
    if X == None:
        X = [0,scale_info['size'][0]]
    if Y == None:
        Y = [0,scale_info['size'][1]]
    if Z == None:
        Z = [0,scale_info['size'][2]] #[0, stackheight]
    
    # Modify upper bound to stay within resolution dimensions
    if X[1] > scale_info['size'][0]:
        X[1] = scale_info['size'][0]
    if Y[1] > scale_info['size'][1]:
        Y[1] = scale_info['size'][1]
    if Z[1] > scale_info['size'][2]:
        Z[1] = scale_info['size'][2]

    # Initialize the output
    datatype = dtype
    image = np.zeros((Y[1]-Y[0],X[1]-X[0],Z[1]-Z[0]),dtype=datatype)

    # If requesting from the lowest scale, then just read the image
    if str(S)==encoder['scales'][0]['key']:
        # image = bfio_reader[Y[0]:Y[1],X[0]:X[1],Z[0]:Z[1],0,0]
        image = vol[Y[0]:Y[1],X[0]:X[1],Z[0]:Z[1]]

        # Encode the chunk
        image_encoded = encode(image)

        # Write the chunk
        slide_writer.store_chunk(image_encoded,str(S),(X[0],X[1],Y[0],Y[1],Z[0],Z[1]))
        logger.info('Scale ({}): {}-{}_{}-{}_{}-{}'.format(S, 
                                                           str(X[0]), str(X[1]), 
                                                           str(Y[0]),str(Y[1]), 
                                                           str(Z[0]), str(Z[1])))
        return image

    else:
        # Set the subgrid dimensions
        subgrid_dims = [[2*X[0],2*X[1]],[2*Y[0],2*Y[1]],[2*Z[0],2*Z[1]]]
        for dim in subgrid_dims:
            while dim[1]-dim[0] > CHUNK_SIZE:
                dim.insert(1,dim[0] + ((dim[1] - dim[0]-1)//CHUNK_SIZE) * CHUNK_SIZE)

        def load_and_scale(*args,**kwargs):
            jutil.attach()
            sub_image = _get_higher_res(**kwargs)
            jutil.detach()
            image = args[0]
            x_ind = args[1]
            y_ind = args[2]
            z_ind = args[3]
            image[y_ind[0]:y_ind[1],x_ind[0]:x_ind[1],z_ind[0]:z_ind[1]] = _mode3(sub_image, datatype)

        with ThreadPoolExecutor(max_workers=8) as executor:
            for x in range(0,len(subgrid_dims[1])-1):
                x_ind = [subgrid_dims[0][x] - subgrid_dims[0][0],subgrid_dims[0][x+1] - subgrid_dims[0][0]]
                x_ind = [np.ceil(xi/2).astype('int') for xi in x_ind]
                for y in range(0,len(subgrid_dims[0])-1):
                    y_ind = [subgrid_dims[1][y] - subgrid_dims[1][0],subgrid_dims[1][y+1] - subgrid_dims[1][0]]
                    y_ind = [np.ceil(yi/2).astype('int') for yi in y_ind]
                    for z in range(0, len(subgrid_dims[2])-1):
                        z_ind = [subgrid_dims[2][z] - subgrid_dims[2][0],subgrid_dims[2][z+1] - subgrid_dims[2][0]]
                        z_ind = [np.ceil(zi/2).astype('int') for zi in z_ind]
                        
                        executor.submit(load_and_scale, 
                                            image, x_ind, y_ind, z_ind, 
                                            X=subgrid_dims[0][x:x+2],
                                            Y=subgrid_dims[1][y:y+2],
                                            Z=subgrid_dims[2][z:z+2],
                                            dtype=dtype,
                                            S=S+1,
                                            vol=vol,
                                            slide_writer=slide_writer,
                                            encoder=encoder)

        # Encode the chunk
        image_encoded = encode(image)
        slide_writer.store_chunk(image_encoded,str(S),(X[0],X[1],Y[0],Y[1],Z[0],Z[1]))
        logger.info('Scale ({}): {}-{}_{}-{}_{}-{}'.format(S, 
                                                           str(X[0]), str(X[1]), 
                                                           str(Y[0]),str(Y[1]), 
                                                           str(Z[0]), str(Z[1])))
        return image