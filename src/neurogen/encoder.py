import trimesh
from . import backend 


def encode_mesh(mesh, compression_level):
    """ Encodes a quantized mesh into Neuroglancer-compatible Draco format

    Parameters
    ----------
    mesh: trimesh.base.Trimesh
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
    vertices: np.ndarray
        An nx3 uint32 numpy array containing quantized vertex coordinates.
    faces: np.ndarray
        An nx3 uint32 numpy array containing mesh faces. 
    compression_level : int
        Level of compression for Draco format from 0 to 10.

    Returns
    -------
    buffer : bytes
        A bytes object containing the encoded mesh.
    """

    return backend.encode_mesh(
            vertices.flatten().astype('uint32'), 
            faces.flatten().astype('uint32'),
            compression_level)
    

