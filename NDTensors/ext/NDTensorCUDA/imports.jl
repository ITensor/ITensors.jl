import NDTensors: set_ndims, set_eltype, to_vector_type, set_eltype_if_unspecified
import NDTensors:
  ContractionProperties, _contract!, GemmBackend, auto_select_backend, _gemm!

import .CUDA: CuArrayAdaptor
