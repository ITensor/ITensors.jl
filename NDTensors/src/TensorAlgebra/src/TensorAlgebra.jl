module TensorAlgebra
using ..NDTensors: Algorithm, @Algorithm_str

include("contract/contract.jl")
include("contract/contract_matricize/contraction_logic.jl")
include("contract/contract_matricize/contract.jl")
end
