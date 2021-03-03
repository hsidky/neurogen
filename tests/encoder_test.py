import os 
import unittest
import numpy as np
import math
import tempfile
import struct
from neurogen import encoder
from neurogen import mesh

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
    
    def test_mesh_generation(self):
        vertices = np.loadtxt(os.path.join(dir_path, 'test_data/sphere_vertices.npy')).astype(np.uint32)
        faces = np.loadtxt(os.path.join(dir_path, 'test_data/sphere_faces.npy')).astype(np.uint32)

        
        # temp_dir = tempfile.TemporaryDirectory()
        with tempfile.TemporaryDirectory() as temp_dir:
            offset_check = 0
            mesh.fulloctree_decomposition(vertices=vertices, 
                                          faces=faces, 
                                          num_lods=3, 
                                          segment_id=1, 
                                          directory=str(temp_dir))
            
            manifest = os.path.join(temp_dir, "meshdir", "1.index")
            eof = os.path.getsize(manifest)
            with open(manifest, 'rb') as manifest_file:
                chunkshapes = list(struct.unpack_from("<3f",manifest_file.read(12)))
                gridorigin = list(struct.unpack_from("<3f",manifest_file.read(12), offset=0))
                num_lods = int(struct.unpack("<I",manifest_file.read(4))[0])
                lod_scales = list((struct.unpack("<"+ str(num_lods)+ "f",manifest_file.read(num_lods*4))))

                self.assertTrue(((vertices.max(axis=0)-vertices.min(axis=0))/(num_lods) == chunkshapes).all())
                self.assertTrue((vertices.min(axis=0)==gridorigin).all())
                self.assertTrue(len(lod_scales)==num_lods)

                for num in range(num_lods):
                    vertex_offsets = list((struct.unpack("<3f", manifest_file.read(12))))
                num_fragments_per_lod = list((struct.unpack("<"+ str(num_lods)+ "I",manifest_file.read(num_lods*4))))
                for i in range(0, num_lods):
                    frags_inthisLOD = int(num_fragments_per_lod[i])
                    if i != (num_lods-1):
                        self.assertTrue(frags_inthisLOD%8==0)
                    fragment_positions_x = list(struct.unpack_from("<" + str(frags_inthisLOD) + "I", manifest_file.read(frags_inthisLOD*4)))
                    fragment_positions_y = list(struct.unpack_from("<" + str(frags_inthisLOD) + "I", manifest_file.read(frags_inthisLOD*4)))
                    fragment_positions_z = list(struct.unpack_from("<" + str(frags_inthisLOD) + "I", manifest_file.read(frags_inthisLOD*4)))
                    fragment_offsets = list(struct.unpack_from("<" + str(frags_inthisLOD)+"I", manifest_file.read(frags_inthisLOD*4)))
                    offset_check = offset_check + sum(fragment_offsets)

                self.assertTrue(eof==manifest_file.tell())

            fragment = os.path.join(temp_dir, "meshdir", "1")
            fragment_file_size = os.path.getsize(fragment)
            self.assertTrue(offset_check==fragment_file_size)



if __name__ == '__main__':
    unittest.main()
