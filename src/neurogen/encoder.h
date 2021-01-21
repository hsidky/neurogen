#include<vector>
#include<cstddef>
#include "draco/compression/decode.h"
#include "draco/compression/encode.h"
#include "draco/core/encoder_buffer.h"
#include "draco/core/vector_d.h"
#include "draco/mesh/triangle_soup_mesh_builder.h"
#include "draco/point_cloud/point_cloud_builder.h"
#include <pybind11/stl.h>

namespace DracoFunctions 
{
    std::string encode_mesh(
        const std::vector<std::uint32_t>& vertices, 
        const std::vector<std::uint32_t>& faces, 
        int compression_level) 
    {
        draco::TriangleSoupMeshBuilder mb;
        mb.Start(faces.size());
        const int pos_att_id = mb.AddAttribute(draco::GeometryAttribute::POSITION, 3, draco::DataType::DT_UINT32);

        for (std::size_t i = 0; i <= faces.size() - 3; i += 3) 
        {
            auto p1i = faces[i]*3;
            auto p2i = faces[i + 1]*3;
            auto p3i = faces[i + 2]*3;
            mb.SetAttributeValuesForFace(
                pos_att_id, 
                draco::FaceIndex(i), 
                draco::Vector3ui(vertices[p1i], vertices[p1i + 1], vertices[p1i + 2]).data(), 
                draco::Vector3ui(vertices[p2i], vertices[p2i + 1], vertices[p2i + 2]).data(), 
                draco::Vector3ui(vertices[p3i], vertices[p3i + 1], vertices[p3i + 2]).data()
            );
        }
        std::unique_ptr<draco::Mesh> mesh = mb.Finalize();
        
        draco::Encoder encoder;        
        draco::EncoderBuffer eb;
        encoder.SetSpeedOptions(10 - compression_level, 10 - compression_level);
        encoder.SetEncodingMethod(draco::MESH_EDGEBREAKER_ENCODING);
        encoder.SetAttributePredictionScheme(draco::GeometryAttribute::POSITION, draco::MESH_PREDICTION_PARALLELOGRAM);
        auto status = encoder.EncodeMeshToBuffer(*mesh, &eb);
        
        return std::string(eb.data(), eb.size());
  }

}

