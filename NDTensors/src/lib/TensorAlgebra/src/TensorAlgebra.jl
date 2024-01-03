module TensorAlgebra
# TODO: Move inside `include`.
using ..AlgorithmSelection: Algorithm, @Algorithm_str
using LinearAlgebra: mul!

include("blockedpermutation.jl")
include("BaseExtensions/BaseExtensions.jl")
include("fusedims.jl")
include("contract/contract.jl")
include("contract/output_labels.jl")
include("contract/blockedperms.jl")
include("contract/allocate_output.jl")
include("contract/contract_matricize/contract.jl")
# TODO: Rename to `TensorAlgebraLinearAlgebraExt`.
include("LinearAlgebraExtensions/LinearAlgebraExtensions.jl")
end
