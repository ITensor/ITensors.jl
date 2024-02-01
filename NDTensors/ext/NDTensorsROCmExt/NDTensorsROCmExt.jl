module NDTensorsROCmExt

using NDTensors
using NDTensors.SetParameters
using NDTensors.Unwrap
using Adapt
using Functors
using LinearAlgebra: LinearAlgebra, Adjoint, Transpose, mul!, svd
using AMDGPU
using AMDGPU.Runtime.Mem
using AMDGPU.rocBLAS
using AMDGPU.rocSOLVER

include("imports.jl")
include("default_kwargs.jl")
include("copyto.jl")
include("set_types.jl")
include("isroc.jl")
include("adapt.jl")
include("indexing.jl")
include("linearalgebra.jl")
include("permutedims.jl")

include("mul.jl")

end
