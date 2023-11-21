using ..NDTensors.NamedDimsArrays: AbstractNamedDimsArray, align
using ..NDTensors.TensorAlgebra: TensorAlgebra
using ..NDTensors: AliasStyle
using ..ITensors: ITensors

function ITensors._contract(na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray)
  return TensorAlgebra.contract(na1, na2)
end

function ITensors._add(na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray)
  return error("Not implemented yet")
end

function ITensors._permute(::AliasStyle, na::AbstractNamedDimsArray, dims::Tuple)
  return align(na, name.(dims))
end

function ITensors._map!!(
  f,
  na_dest::AbstractNamedDimsArray,
  na1::AbstractNamedDimsArray,
  na2::AbstractNamedDimsArray,
)
  return error("Not implemented yet")
end
