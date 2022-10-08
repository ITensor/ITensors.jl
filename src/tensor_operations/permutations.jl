"""
    permute(T::ITensor, inds...; allow_alias = false)

Return a new ITensor `T` with indices permuted according
to the input indices `inds`. The storage of the ITensor
is permuted accordingly.

If called with `allow_alias = true`, it avoids
copying data if possible. Therefore, it may return an alias
of the input ITensor (an ITensor that shares the same data),
such as if the permutation turns out to be trivial.

By default, `allow_alias = false`, and it never
returns an alias of the input ITensor.

# Examples

```julia
i = Index(2, "index_i"); j = Index(4, "index_j"); k = Index(3, "index_k");
T = randomITensor(i, j, k)

pT_1 = permute(T, k, i, j)
pT_2 = permute(T, j, i, k)

pT_noalias_1 = permute(T, i, j, k)
pT_noalias_1[1, 1, 1] = 12
T[1, 1, 1] != pT_noalias_1[1, 1, 1]

pT_noalias_2 = permute(T, i, j, k; allow_alias = false)
pT_noalias_2[1, 1, 1] = 12
T[1, 1, 1] != pT_noalias_1[1, 1, 1]

pT_alias = permute(T, i, j, k; allow_alias = true)
pT_alias[1, 1, 1] = 12
T[1, 1, 1] == pT_alias[1, 1, 1]
```
"""
function permute(T::ITensor, new_inds...; kwargs...)
  if !hassameinds(T, indices(new_inds...))
    error(
      "In `permute(::ITensor, inds...)`, the input ITensor has indices: \n\n$(inds(T))\n\nbut the desired Index ordering is: \n\n$(indices(new_inds...))",
    )
  end
  allow_alias = deprecated_keyword_argument(
    Bool,
    kwargs;
    new_kw=:allow_alias,
    old_kw=:always_copy,
    default=false,
    funcsym=:permute,
    map=!,
  )
  aliasstyle::Union{AllowAlias,NeverAlias} = allow_alias ? AllowAlias() : NeverAlias()
  return permute(aliasstyle, T, new_inds...)
end

# TODO: move to NDTensors
function permutedims(::AllowAlias, T::Tensor, perm)
  return NDTensors.is_trivial_permutation(perm) ? T : permutedims(NeverAlias(), T, perm)
end

# TODO: move to NDTensors, define `permutedims` in terms of `NeverAlias`
function permutedims(::NeverAlias, T::Tensor, perm)
  return permutedims(T, perm)
end

function _permute(as::AliasStyle, T::Tensor, new_inds)
  perm = NDTensors.getperm(new_inds, inds(T))
  return permutedims(as, T, perm)
end

function permute(as::AliasStyle, T::ITensor, new_inds)
  return itensor(_permute(as, tensor(T), new_inds))
end

# Version listing indices
function permute(as::AliasStyle, T::ITensor, new_inds::Index...)
  return permute(as, T, new_inds)
end

"""
    transpose(T::ITensor)

Treating an ITensor as a map from a set of indices
of prime level 0 to a matching set of indices but
of prime level 1
[for example: (i,j,k,...) -> (j',i',k',...)]
return the ITensor which is the transpose of this map.
"""
transpose(T::ITensor) = swapprime(T, 0 => 1)

"""
    ishermitian(T::ITensor; kwargs...)

Test whether an ITensor is a Hermitian operator,
that is whether taking `dag` of the ITensor and
transposing its indices returns numerically
the same ITensor.
"""
function ishermitian(T::ITensor; kwargs...)
  return isapprox(T, dag(transpose(T)); kwargs...)
end

"""
    adjoint(A::ITensor)

For `A'` notation to prime an ITensor by 1.
"""
adjoint(A::ITensor) = prime(A)

#######################################################################
#
# Developer ITensor functions
#

"""
    array(T::ITensor)

Given an ITensor `T`, returns
an Array with a copy of the ITensor's elements,
or a view in the case the the ITensor's storage is Dense.

The ordering of the elements in the Array, in
terms of which Index is treated as the row versus
column, depends on the internal layout of the ITensor.

!!! warning
    This method is intended for developer use
    only and not recommended for use in ITensor applications
    unless you know what you are doing (for example
    you are certain of the memory ordering of the ITensor
    because you permuted the indices into a certain order).

See also [`matrix`](@ref), [`vector`](@ref).
"""
array(T::ITensor) = array(tensor(T))

"""
    array(T::ITensor, inds...)

Convert an ITensor `T` to an Array.

The ordering of the elements in the Array are specified
by the input indices `inds`. This tries to avoid copying
of possible (i.e. may return a view of the original
data), for example if the ITensor's storage is Dense
and the indices are already in the specified ordering
so that no permutation is required.

!!! warning
    Note that in the future we may return specialized
    AbstractArray types for certain storage types,
    for example a `LinearAlgebra.Diagonal` type for
    an ITensor with `Diag` storage. The specific storage
    type shouldn't be relied upon.

See also [`matrix`](@ref), [`vector`](@ref).
"""
array(T::ITensor, inds...) = array(permute(T, inds...; allow_alias=true))

"""
    matrix(T::ITensor)

Given an ITensor `T` with two indices, returns
a Matrix with a copy of the ITensor's elements,
or a view in the case the ITensor's storage is Dense.

The ordering of the elements in the Matrix, in
terms of which Index is treated as the row versus
column, depends on the internal layout of the ITensor.

!!! warning
    This method is intended for developer use
    only and not recommended for use in ITensor applications
    unless you know what you are doing (for example
    you are certain of the memory ordering of the ITensor
    because you permuted the indices into a certain order).

See also [`array`](@ref), [`vector`](@ref).
"""
function matrix(T::ITensor)
  ndims(T) != 2 && throw(DimensionMismatch())
  return array(tensor(T))
end

"""
    matrix(T::ITensor, inds...)

Convert an ITensor `T` to a Matrix.

The ordering of the elements in the Matrix are specified
by the input indices `inds`. This tries to avoid copying
of possible (i.e. may return a view of the original
data), for example if the ITensor's storage is Dense
and the indices are already in the specified ordering
so that no permutation is required.

!!! warning
    Note that in the future we may return specialized
    AbstractArray types for certain storage types,
    for example a `LinearAlgebra.Diagonal` type for
    an ITensor with `Diag` storage. The specific storage
    type shouldn't be relied upon.

See also [`array`](@ref), [`vector`](@ref).
"""
matrix(T::ITensor, inds...) = matrix(permute(T, inds...; allow_alias=true))

"""
    vector(T::ITensor)

Given an ITensor `T` with one index, returns
a Vector with a copy of the ITensor's elements,
or a view in the case the ITensor's storage is Dense.

See also [`array`](@ref), [`matrix`](@ref).
"""
function vector(T::ITensor)
  ndims(T) != 1 && throw(DimensionMismatch())
  return array(tensor(T))
end

"""
    vector(T::ITensor, inds...)

Convert an ITensor `T` to an Vector.

The ordering of the elements in the Array are specified
by the input indices `inds`. This tries to avoid copying
of possible (i.e. may return a view of the original
data), for example if the ITensor's storage is Dense
and the indices are already in the specified ordering
so that no permutation is required.

!!! warning
    Note that in the future we may return specialized
    AbstractArray types for certain storage types,
    for example a `LinearAlgebra.Diagonal` type for
    an ITensor with `Diag` storage. The specific storage
    type shouldn't be relied upon.

See also [`array`](@ref), [`matrix`](@ref).
"""
vector(T::ITensor, inds...) = vector(permute(T, inds...; allow_alias=true))
