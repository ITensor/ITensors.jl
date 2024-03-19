module NDTensorsAMDGPUExt

using NDTensors
using Adapt
using Functors
using AMDGPU
using AMDGPU.Runtime.Mem
using AMDGPU.rocBLAS
using AMDGPU.rocSOLVER

include("copyto.jl")
include("set_types.jl")
include("adapt.jl")
include("indexing.jl")
include("linearalgebra.jl")
include("mul.jl")
include("permutedims.jl")

end
