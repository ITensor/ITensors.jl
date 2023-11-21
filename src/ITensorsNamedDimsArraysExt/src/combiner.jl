# Combiner
using ..NDTensors.NamedDimsArrays: AbstractNamedDimsArray
# using ..NDTensors.TensorAlgebra: fusedims, unfusedims
using NDTensors: Tensor, Combiner
function ITensors._contract(na::AbstractNamedDimsArray, c::Tensor{<:Any,<:Any,<:Combiner})
  error("NI")
  ## fusedims(na, is => c)
  ## unfusedims(na, c => is)
end
