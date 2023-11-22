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
  # TODO: Handle permutation better.
  a1 = unname(na1)
  a2 = unname(na2, dimnames(na1))
  return named(a1 + a2, dimnames(na1))
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
  # TODO: Handle permutations better!
  # TODO: Preserve names in `permutedims`.
  a1 = unname(na1, dimnames(na_dest))
  a2 = unname(na2, dimnames(na_dest))
  a_dest = unname(na_dest)
  map!(f, a_dest, a1, a2)
  return na_dest
end
