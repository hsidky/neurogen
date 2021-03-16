import os 
import unittest
import numpy as np
import math
import tempfile
import struct
import imageio

from neurogen import encoder
from neurogen import mesh as ngmesh
from neurogen import info as nginfo
from neurogen import volume as ngvol

from copy import deepcopy

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
    
    def test_info_file_specification(self):
        size, radius = 100, 20
        A = np.zeros((size,size,size))
        volume = deepcopy(A) 
        x0, y0, z0 = int(np.floor(A.shape[0]/2)), \
                    int(np.floor(A.shape[1]/2)), int(np.floor(A.shape[2]/2))
        for x in range(x0-radius, x0+radius+1):
            for y in range(y0-radius, y0+radius+1):
                for z in range(z0-radius, z0+radius+1):
                    deb = radius - abs(x0-x) - abs(y0-y) - abs(z0-z) 
                    if (deb)>=0: 
                        volume[x,y,z] = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            info_dict = nginfo.info_image(directory=str(temp_dir),
                                          dtype=volume.dtype,
                                          chunk_size=[64,64,64],
                                          size=volume.shape)
            scales = info_dict['scales']
            num_scales = len(scales)
            self.assertTrue(info_dict['scales'][-1]['size'] == [1,1,1])

            encodevolume = ngvol.generate_recursive_chunked_representation(volume=volume,
                                                                           info=info_dict,
                                                                           dtype=volume.dtype,
                                                                           directory=temp_dir,
                                                                           blurring_method='average')
            for i in range(num_scales):
                scale_key = scales[i]['key']
                scale_dir = os.path.join(str(temp_dir), scale_key)
                scale_size = scales[i]['size']
                num_directories = len(os.listdir(scale_dir))
                xfiles = np.ceil(scale_size[0]/64)
                yfiles = np.ceil(scale_size[1]/64)
                zfiles = np.ceil(scale_size[2]/64)
                self.assertTrue(xfiles*yfiles*zfiles == num_directories)

    def test_mesh_generation(self):
        vertices = np.loadtxt(os.path.join(dir_path, 'test_data/sphere_vertices.npy')).astype(np.uint32)
        faces = np.loadtxt(os.path.join(dir_path, 'test_data/sphere_faces.npy')).astype(np.uint32)

        # temp_dir = tempfile.TemporaryDirectory()
        with tempfile.TemporaryDirectory() as temp_dir:
            offset_check = 0
            ngmesh.fulloctree_decomposition(vertices=vertices, 
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
