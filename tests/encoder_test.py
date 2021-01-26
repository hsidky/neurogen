import os 
import unittest
import numpy as np
from neurogen import encoder 

dir_path = os.path.dirname(os.path.realpath(__file__))

class TestEncodingDecoding(unittest.TestCase):
    def test_encode_decode(self):
        vertices = np.loadtxt(os.path.join(dir_path, 'test_data/sphere_vertices.npy')).astype(np.uint32)
        faces = np.loadtxt(os.path.join(dir_path, 'test_data/sphere_faces.npy')).astype(np.uint32)

        buffer = encoder.encode_vertices_faces(vertices, faces, compression_level=5)
        vertices_decoded, faces_decoded = encoder.decode_buffer(buffer)

        self.assertTrue(
            np.array_equal(
                np.sort(vertices[faces].flat), 
                np.sort(vertices_decoded[faces_decoded].flat)
            )
        )


if __name__ == '__main__':
    unittest.main()