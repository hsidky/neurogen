import trimesh
import numpy as np
import pandas as pd
import os, struct, json
import encoder

class Quantize():
    """
    A class used to quantize mesh vertex positions for Neuroglancer precomputed
    meshes to a specified number of bits.
    
    Based on the C++ code provided here: https://github.com/google/neuroglancer/issues/266#issuecomment-739601142

    Attributes
    ----------
    upper_bound : int 
        The largest integer used to represent a vertex position.
    scale : np.ndarray
        Array containing the scaling factors for each dimension. 
    offset : np.ndarray
        Array containing the offset values for each dimension. 
    """

    def __init__(self, fragment_origin, fragment_shape, input_origin, quantization_bits):
        """
        Parameters
        ----------
        fragment_origin : np.ndarray
            Minimum input vertex position to represent.
        fragment_shape : np.ndarray
            The inclusive maximum vertex position to represent is `fragment_origin + fragment_shape`.
        input_origin : np.ndarray
            The offset to add to input vertices before quantizing them.
        quantization_bits : int
            The number of bits to use for quantization.
        """
        self.upper_bound = np.iinfo(np.uint32).max >> (np.dtype(np.uint32).itemsize*8 - quantization_bits)
        self.scale = self.upper_bound / fragment_shape
        self.offset = input_origin - fragment_origin + 0.5/self.scale
    
    def __call__(self, vertices):
        """ Quantizes an Nx3 numpy array of vertex positions.
        
        Parameters
        ----------
        vertices : np.ndarray
            Nx3 numpy array of vertex positions.
        
        Returns
        -------
        np.ndarray
            Quantized vertex positions.
        """
        output = np.minimum(self.upper_bound, np.maximum(0, self.scale*(vertices + self.offset))).astype(np.uint32)
        return output

def cmp_zorder(lhs, rhs):
    """Compare z-ordering
    
    Code taken from https://en.wikipedia.org/wiki/Z-order_curve
    """
    def less_msb(x: int, y: int):
        return x < y and x < (x ^ y)

    # Assume lhs and rhs array-like objects of indices.
    assert len(lhs) == len(rhs)
    # Will contain the most significant dimension.
    msd = 2
    # Loop over the other dimensions.
    for dim in [1, 0]:
        # Check if the current dimension is more significant
        # by comparing the most significant bits.
        if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
            msd = dim
    return lhs[msd] - rhs[msd]

def clean_mesh(mesh):
    """This function cleans up the mesh for decimating the mesh.
    
    Returns
    -------
    mesh : trimesh.base.Trimesh
        A mesh that has been "cleaned" so that it can be manipulated for LODS
    """

    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.fill_holes()

    return mesh

def scale_mesh(mesh, scale):
    """ This function scales the vertices to range from 0 to scale 
    
    Parameters
    ----------
    mesh : trimesh.base.Trimesh
        A Trimesh mesh object to scale
    scale : int
        Specifies the max for the new range
    
    Returns
    -------
    scaled_mesh : trimesh.base.Trimesh
        Trimesh mesh object whose vertices ranges from 0 to scale
    """

    
    vertices = mesh.vertices
    maxval = vertices.max(axis=0)
    minval = vertices.min(axis=0)

    max_nodes = scale/(maxval-minval)
    verts_scaled = max_nodes*(vertices - minval)
    scaled_mesh = mesh.copy()
    scaled_mesh.vertices = verts_scaled

    return scaled_mesh

def fulloctree_decomposition(vertices,
                            faces,
                            num_lods, 
                            segment_id,
                            directory,
                            quantization_bits=16,
                            compression_level=5,
                            mesh_subdirectory='meshdir'):

    """ Generates a Neuroglancer precomputed multiresolution mesh.
    Parameters
    ----------
    vertices : numpy array
        Vertices to convert to trimesh object.
    faces : numpy array
        Faces to convert to trimesh object.
    num_lods : int
        Number of levels of detail to generate.
    segment_id : str
        The ID of the segment to which the mesh belongs. 
    directory : str
        Neuroglancer precomputed volume directory.
    quantization_bits : int
        Number of bits for mesh vertex quantization. Can only be 10 or 16. 
    compression_level : int
        Level of compression for Draco format.
    mesh_subdirectory : str
        Name of the mesh subdirectory within the Neuroglancer volume directory.
    """


    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    fulloctree_decompositio_mesh(mesh=mesh,
                                 num_lods=num_lods,
                                 segment_id=segment_id,
                                 directory=directory,
                                 quantization_bits=quantization_bits,
                                 compression_level=compression_level,
                                 mesh_subdirectory=mesh_subdirectory)

def fulloctree_decompositio_mesh(mesh,
                             num_lods, 
                             segment_id,
                             directory,
                             quantization_bits=16,
                             compression_level=5,
                             mesh_subdirectory='meshdir'):
    """ Generates a Neuroglancer precomputed multiresolution mesh.
    
    Parameters
    ----------
    mesh : trimesh.base.Trimesh 
        A Trimesh mesh object to decompose.
    num_lods : int
        Number of levels of detail to generate.
    segment_id : str
        The ID of the segment to which the mesh belongs. 
    directory : str
        Neuroglancer precomputed volume directory.
    quantization_bits : int
        Number of bits for mesh vertex quantization. Can only be 10 or 16. 
    compression_level : int
        Level of compression for Draco format.
    mesh_subdirectory : str
        Name of the mesh subdirectory within the Neuroglancer volume directory.
    """
    # Raise Error if quantization bits does not equal 10 or 16
    if (quantization_bits != 10) or (quantization_bits != 16):
        raise ValueError('Quantization bits must be 10 or 16 to be compatible with Neuroglancer')
    
    # Mesh values
    mesh_vertices = mesh.vertices
    max_mesh_vertex = mesh_vertices.max(axis=0)
    min_mesh_vertex = mesh_vertices.min(axis=0)
    clean_mesh(mesh)

    # Initialize Arrays used to define the decomposition
    lods = np.arange(0, num_lods)
    scales = np.power(2, lods)

    # For each LOD, define how much the mesh is going to be simplified 
        # by reducing the number of faces
    decimate_by = np.power(np.true_divide(num_lods,scales),2)
    num_faces = mesh.faces.shape[0] 
    num_faces_left = num_faces//decimate_by

    # Create directory
    mesh_dir = os.path.join(directory, mesh_subdirectory)
    os.makedirs(mesh_dir, exist_ok=True)

    # Write manifest file/fragment file as specified:
    # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md
    chunk_shape = (max_mesh_vertex - min_mesh_vertex)/(2**num_lods)
    grid_origin = min_mesh_vertex
    vertex_offsets = np.array([[0., 0., 0.] for _ in range(num_lods)])
    num_fragments_per_lod = np.flip(np.power(8, lods))
    manifest_filename = os.path.join(mesh_dir, f'{segment_id}.index')
    with open(manifest_filename, 'ab') as manifest_file:
        manifest_file.write(chunk_shape.astype('<f').tobytes())
        manifest_file.write(grid_origin.astype('<f').tobytes())
        manifest_file.write(struct.pack('<I', num_lods))
        manifest_file.write(scales.astype('<f').tobytes())
        manifest_file.write(vertex_offsets.astype('<f').tobytes(order='C'))
        manifest_file.write(num_fragments_per_lod.astype('<I').tobytes())

        # Write fragment file
        with open(os.path.join(mesh_dir, f'{segment_id}'), 'wb') as fragment_file:
            
            for i in reversed(lods):

                decimated_mesh = mesh.simplify_quadratic_decimation(num_faces_left[i])
                clean_mesh(decimated_mesh)

                nodes_per_dim = scales[i]

                # The vertices need to range from 0 to number of nodes in mesh
                scaled_mesh = scale_mesh(decimated_mesh, nodes_per_dim)

                # Define plane normals and scale mesh.
                nyz, nxz, nxy = np.eye(3)

                # Variables that will be appended to the manifest file
                lod_pos = []
                lod_off = []

                # The mesh gets sliced at every node
                for x in range(0, nodes_per_dim):
                    mesh_x = trimesh.intersections.slice_mesh_plane(scaled_mesh, plane_normal=nyz, plane_origin=nyz*x)
                    mesh_x = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=-nyz, plane_origin=nyz*(x+1))
                    for y in range(0, nodes_per_dim):
                        mesh_y = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=nxz, plane_origin=nxz*y)
                        mesh_y = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=-nxz, plane_origin=nxz*(y+1))
                        for z in range(0, nodes_per_dim):
                            mesh_z = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=nxy, plane_origin=nxy*z)
                            mesh_z = trimesh.intersections.slice_mesh_plane(mesh_z, plane_normal=-nxy, plane_origin=nxy*(z+1))

                            # Initialize Quantizer.
                            quantizer = Quantize(
                                fragment_origin=np.array([x, y, z]), 
                                fragment_shape=np.array([1, 1, 1]), 
                                input_origin=np.array([0,0,0]), 
                                quantization_bits=quantization_bits
                            )

                            dracolen = 0
                            if len(mesh_z.vertices) > 0:
                                mesh_z.vertices = quantizer(mesh_z.vertices)
                                draco = encoder.encode_mesh(mesh_z,compression_level=compression_level)
                                
                                dracolen = len(draco)
                                fragment_file.write(draco)


                            lod_off.append(dracolen)
                            lod_pos.append([x, y, z])

                manifest_file.write(np.array(lod_pos).T.astype('<I').tobytes(order='C'))
                manifest_file.write(np.array(lod_off).astype('<I').tobytes(order='C'))

    manifest_file.close()
    fragment_file.close()

def density_decomposition(vertices,
                        faces,
                        segment_id,
                        directory,
                        minimum_vertices=1024,
                        quantization_bits=16,
                        compression_level=5,
                        mesh_subdirectory='meshdir'):

    """ Generates a Neuroglancer precomputed multiresolution mesh based
    on the number of vertices within each fragment
    
    Parameters
    ----------
    vertices : numpy array
        Vertices to convert to trimesh object.
    faces : numpy array
        Faces to convert to trimesh object.
    segment_id : str
        The ID of the segment to which the mesh belongs. 
    directory : str
        Neuroglancer precomputed volume directory.
    minimum : int
        The minimum number of vertices that a fragment needs to have in order a
        stop breaking into more octrees.  Default is 1024
    quantization_bits : int
        Number of bits for mesh vertex quantization. Can only be 10 or 16. 
    compression_level : int
        Level of compression for Draco format.
    mesh_subdirectory : str
        Name of the mesh subdirectory within the Neuroglancer volume directory.    
    """

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    density_decomposition_mesh(mesh=mesh,
                               segment_id=segment_id,
                               minimum_vertices=minimum_vertices,
                               quantization_bits=quantization_bits,
                               compression_level=compression_level,
                               mesh_subdirectory=mesh_subdirectory)

def generate_mesh_dataframe(vertices, minvertices, lod=0):

    """ This function generates a dataframe that contains information 
    on the levels of details based on density.  The function bin the vertices 
    to fit in the number nodes in the respective LOD. It will continue until
    all fragments of the highest level of detail contains less than minvertices.
    
    Parameters
    ----------
    vertices : numpy array 
        A 3x1 array of the mesh's vertices
    minvertices : int
        The maximum number of vertices within a fragment in 
        the highest level of detail
    lod : int
        The current number of levels to create the mesh

    Returns
    -------
    lod : int
        The number of levels created for the mesh
    mesh_lods : pandas dataframe
        The dataframe containing information on the level of 
        details
    """
    # Intialize necessary information
    reachlevel = False 
    num_vertices = len(vertices)

    # Initalize necessary information to prevent recalculating
    maxvertex = vertices.max(axis=0)
    minvertex = vertices.min(axis=0)

    # Initialize the dataframes
    # mesh_df contains the orginial vertices of the mesh
    mesh_df = pd.DataFrame(data=vertices, columns=["x", "y", "z"])
    # mesh_lods contains information of all the level of details
    mesh_lods = pd.DataFrame(data=[[0, 0, 0, num_vertices, 0, None]], columns = "x y z Count LOD Decompose".split())
    # mesh_lod will only contains information on the current level of detail
        # will continuously get updated
    mesh_lod = pd.DataFrame(data=[], columns=["x", "y", "z", "Decompose"])

    while reachlevel == False:
        # define the number of nodes, the number of times the mesh gets sliced 
            # in the level of detail
        numsplits = int((2**lod)+1)
        xsplits = np.linspace(start=minvertex[0], stop=maxvertex[0], num=numsplits)
        ysplits = np.linspace(start=minvertex[1], stop=maxvertex[1], num=numsplits)
        zsplits = np.linspace(start=minvertex[2], stop=maxvertex[2], num=numsplits)

        # meshnodes_df will be the binned data of mesh_df
        meshnodes_df = mesh_df.copy()
        for x in range(numsplits-1):
            for y in range(numsplits-1):
                for z in range(numsplits-1):
                    
                    # if the (x,y,z) coordinates are within that range then they get binned
                        # into that fragment
                    condx = (mesh_df["x"] >= xsplits[x]) & (mesh_df["x"] < xsplits[x+1])
                    condy = (mesh_df["y"] >= ysplits[y]) & (mesh_df["y"] < ysplits[y+1])
                    condz = (mesh_df["z"] >= zsplits[z]) & (mesh_df["z"] < zsplits[z+1])

                    # This section looks at the previous level of detail.
                        # If the previous level specifies that there is no need
                        # to decompose that fragment, because it is less than 
                        # the minimum_vertices, then it replaces those vertices 
                        # in the next level of detail with empty values.
                    if not mesh_lod[(mesh_lod["x"] == x//2) & 
                                    (mesh_lod["y"] == y//2) &
                                    (mesh_lod["z"] == z//2) &
                                    (mesh_lod["Decompose"] == False)].empty:
                        meshnodes_df.loc[condx,"x"] = None
                        meshnodes_df.loc[condy,"y"] = None
                        meshnodes_df.loc[condz,"z"] = None
                    else:
                        meshnodes_df.loc[condx,"x"] = x
                        meshnodes_df.loc[condy,"y"] = y
                        meshnodes_df.loc[condz,"z"] = z

        # The extents of the mesh needs to be within the last fragment
        meshnodes_df.loc[mesh_df["x"]==xsplits[-1],"x"] = x
        meshnodes_df.loc[mesh_df["y"]==ysplits[-1],"y"] = y
        meshnodes_df.loc[mesh_df["z"]==zsplits[-1],"z"] = z

        # Count the binned data
        mesh_lod = meshnodes_df.groupby(["x", "y", "z"]).size().reset_index(name="Count")
        mesh_lod["LOD"] = lod
        mesh_lod["Decompose"] = np.where(mesh_lod["Count"] > minvertices, True, False)
        
        # Append this level of details information to the output dataframe
        if lod == 0:
            mesh_lods = mesh_lod
        else:
            mesh_lods = mesh_lods.append(mesh_lod)
        
        # If all the vertices are less than minimum_vertices, then stop 
            # the while loop
        lod = lod+1
        stop_condition = (mesh_lod["Decompose"] == False)

        if stop_condition.all():
            reachlevel = True
    mesh_lods = mesh_lods.reset_index(drop=True)

    return lod, mesh_lods


def density_decomposition_mesh(mesh,
                        segment_id,
                        directory,
                        minimum_vertices=1024,
                        quantization_bits=16,
                        compression_level=5,
                        mesh_subdirectory='meshdir'):

    """ Generates a Neuroglancer precomputed multiresolution mesh based
    on the number of vertices within each fragment
    
    Parameters
    ----------
    mesh : trimesh.base.Trimesh 
        A Trimesh mesh object to decompose.
    segment_id : str
        The ID of the segment to which the mesh belongs. 
    directory : str
        Neuroglancer precomputed volume directory.
    minimum : int
        The minimum number of vertices that a fragment needs to have in order a
        stop breaking into more octrees.  Default is 1024
    quantization_bits : int
        Number of bits for mesh vertex quantization. Can only be 10 or 16. 
    compression_level : int
        Level of compression for Draco format.
    mesh_subdirectory : str
        Name of the mesh subdirectory within the Neuroglancer volume directory.    
    """
    
    # Mesh values
    mesh_vertices = mesh.vertices
    max_mesh_vertex = mesh_vertices.max(axis=0)
    min_mesh_vertex = mesh_vertices.min(axis=0)
    clean_mesh(mesh)

    # Need to get information on meshes LOD prior to decimation.
    num_lods, dataframe = generate_mesh_dataframe(vertices=mesh.vertices, 
                                                                    minvertices=minimum_vertices,
                                                                    lod=0)

    # Initialize Arrays used to define the decomposition
    lods = np.arange(0, num_lods)
    scales = np.power(2, lods)

    # For each LOD, define how much the mesh is going to be simplified 
        # by reducing the number of faces
    decimate_by = np.power(np.true_divide(num_lods,scales),2)
    num_faces = mesh.faces.shape[0] 
    num_faces_left = num_faces//decimate_by

    # Create directory
    mesh_dir = os.path.join(directory, mesh_subdirectory)
    os.makedirs(mesh_dir, exist_ok=True)

    # Write manifest file/fragment file as specified:
    # https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md
    chunk_shape = (max_mesh_vertex - min_mesh_vertex)/(2**num_lods)
    grid_origin = min_mesh_vertex
    vertex_offsets = np.array([[0., 0., 0.] for _ in range(num_lods)])
    num_fragments_per_lod = np.flip(np.array(dataframe["LOD"].value_counts(sort=False).to_list()))
    manifest_filename = os.path.join(mesh_dir, f'{segment_id}.index')
    with open(manifest_filename, 'ab') as manifest_file:
        manifest_file.write(chunk_shape.astype('<f').tobytes())
        manifest_file.write(grid_origin.astype('<f').tobytes())
        manifest_file.write(struct.pack('<I', num_lods))
        manifest_file.write(scales.astype('<f').tobytes())
        manifest_file.write(vertex_offsets.astype('<f').tobytes(order='C'))
        manifest_file.write(num_fragments_per_lod.astype('<I').tobytes())

        # Write fragment file.
        with open(os.path.join(mesh_dir, f'{segment_id}'), 'wb') as fragment_file:
            
            for i in reversed(lods):

                lodframe = dataframe[(dataframe["LOD"] == i)]
                decimated_mesh = mesh.simplify_quadratic_decimation(num_faces_left[i])
                clean_mesh(decimated_mesh)

                # The scale is the same as the number of nodes in each dimension.
                nodes_per_dim = scales[i]

                # The vertices need to range from 0 to number of nodes in mesh
                scaled_mesh = scale_mesh(decimated_mesh, nodes_per_dim)

                # Define plane normals and scale mesh.
                nyz, nxz, nxy = np.eye(3)

                # Variables that will be appended to the manifest file
                lod_pos = []
                lod_off = []

                # The mesh gets sliced at every node
                for x in range(0, nodes_per_dim):
                    mesh_x = trimesh.intersections.slice_mesh_plane(scaled_mesh, plane_normal=nyz, plane_origin=nyz*x)
                    mesh_x = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=-nyz, plane_origin=nyz*(x+1))
                    for y in range(0, nodes_per_dim):
                        mesh_y = trimesh.intersections.slice_mesh_plane(mesh_x, plane_normal=nxz, plane_origin=nxz*y)
                        mesh_y = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=-nxz, plane_origin=nxz*(y+1))
                        for z in range(0, nodes_per_dim):
                            mesh_z = trimesh.intersections.slice_mesh_plane(mesh_y, plane_normal=nxy, plane_origin=nxy*z)
                            mesh_z = trimesh.intersections.slice_mesh_plane(mesh_z, plane_normal=-nxy, plane_origin=nxy*(z+1))
                            
                            condx = (lodframe["x"] == x)
                            condy = (lodframe["y"] == y)
                            condz = (lodframe["z"] == z)

                            # Information must be present in dataframe
                            if (condx & condy & condz).any():

                                # Initialize Quantizer.
                                quantizer = Quantize(
                                    fragment_origin=np.array([x, y, z]), 
                                    fragment_shape=np.array([1, 1, 1]), 
                                    input_origin=np.array([0,0,0]), 
                                    quantization_bits=quantization_bits
                                )
                                dracolen = 0 # Append zero if fragment contains no vertices
                                if (len(mesh_z.vertices) > 0):
                                        mesh_z.vertices = quantizer(mesh_z.vertices)
                                        draco = encoder.encode_mesh(mesh_z,compression_level=compression_level)
                                        dracolen = len(draco)

                                        fragment_file.write(draco)
 
                                lod_off.append(dracolen)
                                lod_pos.append([x,y,z])

                manifest_file.write(np.array(lod_pos).T.astype('<I').tobytes(order='C'))
                manifest_file.write(np.array(lod_off).astype('<I').tobytes(order='C'))
    
    fragment_file.close()
    manifest_file.close()


