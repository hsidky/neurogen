#include <pybind11/pybind11.h>
#include "encoder.h"

namespace py = pybind11;

PYBIND11_MODULE(backend, m) 
{
    m.def("encode_mesh", [](
        const std::vector<std::uint32_t>& vertices, 
        const std::vector<std::uint32_t>& faces, 
        int compression){
            auto s = DracoFunctions::encode_mesh(vertices, faces, compression);
            return py::bytes(s);
    });
}