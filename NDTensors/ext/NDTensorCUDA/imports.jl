import NDTensors: set_ndims, set_eltype, set_eltype_if_unspecified
import NDTensors:
  ContractionProperties, _contract!, GemmBackend, auto_select_backend, _gemm!

import .CUDA: CuArrayAdaptor
