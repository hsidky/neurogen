import trimesh
import numpy as np
import os, struct, json
from neurogen import backend 

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

def encode_mesh(mesh, compression_level):
    """ Encodes a quantized mesh into Neuroglancer-compatible Draco format

    Parameters
    ----------
    mesh : trimesh.base.Trimesh
        A Trimesh mesh object to encode
    compression_level : int
        Level of compression for Draco format from 0 to 10.

    Returns
    -------
    buffer : bytes
        A bytes object containing the encoded mesh.
    """

    return encode_vertices_faces(mesh.vertices, mesh.faces, compression_level)



def encode_vertices_faces(vertices, faces, compression_level):
    """ Encodes a set of quantized vertices and faces into 
        Neuroglancer-compatible Draco format

    Parameters
    ----------
    vertices : np.ndarray
        An nx3 uint32 numpy array containing quantized vertex coordinates.
    faces : np.ndarray
        An nx3 uint32 numpy array containing mesh faces. 
    compression_level : int
        Level of compression for Draco format from 0 to 10.

    Returns
    -------
    buffer : bytes
        A bytes object containing the encoded mesh.
    """

    return backend.encode_mesh(
            vertices.flatten().astype(np.uint32), 
            faces.flatten().astype(np.uint32),
            compression_level)


def decode_buffer(buffer):
    """ Decodes Draco buffer into vertices and faces

    Parameters
    ----------
    buffer : bytes
        A bytes object containing a Draco mesh buffer.
    
    Returns
    -------
    vertices : np.ndarray
        An nx3 uint32 numpy array containing quantized vertex coordinates.
    faces : np.ndarray
        An nx3 uint32 numpy array containing mesh faces.
    """

    vertices, faces = backend.decode_mesh(buffer)

    vertices = np.asarray(vertices, dtype=np.uint32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)

    return vertices, faces

def clean_mesh(mesh):
    "This function cleans up the mesh."

    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.fill_holes()

    return mesh

def scale_mesh(mesh, scale):

    vertices = mesh.vertices
    maxval = vertices.max(axis=0)
    minval = vertices.min(axis=0)

    max_nodes = scale/(maxval-minval)
    verts_scaled = max_nodes*(vertices - minval)
    scaled_mesh = mesh.copy()
    scaled_mesh.vertices = verts_scaled

    return scaled_mesh

def fulloctree_decomposition(mesh,
                             num_lods, 
                             segment_id,
                             directory,
                             scale_by=2,
                             quantization_bits=16,
                             compression_level=5):
    
    # Mesh values
    mesh_vertices = mesh.vertices
    max_mesh_vertex = mesh_vertices.max(axis=0)
    min_mesh_vertex = mesh_vertices.min(axis=0)

    # Initialize Arrays used to define the decomposition
    lods = np.arange(0, num_lods)
    scales = np.power(scale_by, lods)

    # For each LOD, define how much the mesh is going to be simplified 
        # by reducing the number of faces
    decimate_by = np.power(np.true_divide(num_lods,scales),scale_by)
    num_faces = mesh.faces.shape[0] 
    num_faces_left = num_faces//decimate_by

    # Create directory
    mesh_dir = os.path.join(directory, 'meshdir')
    os.makedirs(mesh_dir, exist_ok=True)

    # Write manifest file.
    chunk_shape = max_mesh_vertex - min_mesh_vertex
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

        # Write fragment file.
        with open(os.path.join(mesh_dir, f'{segment_id}'), 'wb') as fragment_file:
            
            for i in reversed(lods):

                decimated_mesh = mesh.simplify_quadratic_decimation(num_faces_left[i])
                clean_mesh(decimated_mesh)

                nodes_per_dim = scales[i]

                # The vertices need to range from 0 to number of nodes in mesh
                scaled_mesh = scale_mesh(decimated_mesh, nodes_per_dim)

                # Define plane normals and scale mesh.
                nyz, nxz, nxy = np.eye(3)

                lod_pos = []
                lod_off = []
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
                                draco = encode_mesh(mesh_z,compression_level=compression_level)
                                fragment_file.write(draco)
                                dracolen = len(draco)

                            lod_off.append(dracolen)
                            lod_pos.append([x, y, z])

                manifest_file.write(np.array(lod_pos).T.astype('<I').tobytes(order='C'))
                manifest_file.write(np.array(lod_off).astype('<I').tobytes(order='C'))

    manifest_file.close()
    fragment_file.close()