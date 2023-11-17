module TensorAlgebra
using LinearAlgebra: mul!
using ..NDTensors: Algorithm, @Algorithm_str

include("bipartitionedpermutation.jl")
include("fusedims.jl")
include("contract/contract.jl")
include("contract/output_labels.jl")
include("contract/allocate_output.jl")
include("contract/contract_matricize/contract.jl")
end
