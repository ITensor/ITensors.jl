module NDTensorsCUDAExt

using NDTensors
using NDTensors.SetParameters
using NDTensors.Unwrap
using Adapt
using Functors
using LinearAlgebra: LinearAlgebra, Adjoint, Transpose, mul!, svd
using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER

## TODO I added copyto and permutedims which match the functions in 
## NDTensorsMetalExt because I found similar issues in CUDA
include("imports.jl")
include("default_kwargs.jl")
include("copyto.jl")
include("set_types.jl")
include("iscu.jl")
include("adapt.jl")
include("indexing.jl")
include("linearalgebra.jl")
include("mul.jl")
include("permutedims.jl")
end
