# AbstractArray algebra needed for ITensors.jl.
# TODO: Instead dispatch on `tensortype(::ITensor)` from within
# ITensors.jl.
using ..NDTensors.NamedDimsArrays: AbstractNamedDimsArray, align
using ..NDTensors.TensorAlgebra: TensorAlgebra
using ..NDTensors: AliasStyle
using ..ITensors: ITensors

function ITensors._contract(na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray)
  return TensorAlgebra.contract(na1, na2)
end

function ITensors._add(na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray)
  return na1 + na2
end

function ITensors._permute(::AliasStyle, na::AbstractNamedDimsArray, dims::Tuple)
  # TODO: Handle aliasing properly.
  return align(na, name.(dims))
end

function ITensors._map!!(
  f,
  na_dest::AbstractNamedDimsArray,
  na1::AbstractNamedDimsArray,
  na2::AbstractNamedDimsArray,
)
  # TODO: Handle maybe-mutation.
  map!(f, na_dest, na1, na2)
  return na_dest
end
