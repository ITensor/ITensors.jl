# AbstractArray algebra
# TODO: Move to `NamedDimsArrays`.
# TODO: Make this more generic, based on `VectorInterface.add!!`,
# broadcast, map, etc.
Base.:*(a::AbstractNamedDimsArray, c::Number) = named(unname(a) * c, dimnames(a))
Base.:*(c::Number, a::AbstractNamedDimsArray) = named(c * unname(a), dimnames(a))

using ..NDTensors.NamedDimsArrays: AbstractNamedDimsArray, align
using ..NDTensors.TensorAlgebra: TensorAlgebra
using ..NDTensors: AliasStyle
using ..ITensors: ITensors

function ITensors._contract(na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray)
  return TensorAlgebra.contract(na1, na2)
end

function ITensors._add(na1::AbstractNamedDimsArray, na2::AbstractNamedDimsArray)
  # TODO: Implement.
  return error("Not implemented yet")
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
  perm1 = map(n -> findfirst(isequal(n), dimnames(na_dest)), dimnames(na1))
  perm2 = map(n -> findfirst(isequal(n), dimnames(na_dest)), dimnames(na2))
  perm1 = invperm(perm1)
  perm2 = invperm(perm2)
  # TODO: Preserve names in `permutedims`.
  na1 = named(permutedims(na1, perm1), map(i -> dimnames(na1)[i], perm1))
  na2 = named(permutedims(na2, perm2), map(i -> dimnames(na2)[i], perm2))
  perm1 = map(n -> findfirst(isequal(n), dimnames(na_dest)), dimnames(na1))
  perm2 = map(n -> findfirst(isequal(n), dimnames(na_dest)), dimnames(na2))
  map!(f, na_dest, na1, na2)
  return na_dest
end
