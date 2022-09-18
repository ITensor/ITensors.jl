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
    dag(T::ITensor; allow_alias = true)

Complex conjugate the elements of the ITensor `T` and dagger the indices.

By default, an alias of the ITensor is returned (i.e. the output ITensor
may share data with the input ITensor). If `allow_alias = false`,
an alias is never returned.
"""

function dag(as::AliasStyle, T::Tensor{ElT,N}) where {ElT,N}
  if using_auto_fermion() && has_fermionic_subspaces(inds(T)) # <fermions>
    CT = conj(NeverAlias(), T)
    NDTensors.scale_blocks!(CT, block -> NDTensors.permfactor(reverse(1:N), block, inds(T)))
    return setinds(CT, dag(inds(T)))
  end
  return setinds(conj(as, T), dag(inds(T)))
end

function dag(as::AliasStyle, T::ITensor)
  return itensor(dag(as, tensor(T)))
end

# Helpful for generic code
dag(x::Number) = conj(x)

function dag(T::ITensor; kwargs...)
  allow_alias::Bool = deprecated_keyword_argument(
    Bool,
    kwargs;
    new_kw=:allow_alias,
    old_kw=:always_copy,
    default=true,
    funcsym=:dag,
    map=!,
  )
  aliasstyle::Union{AllowAlias,NeverAlias} = allow_alias ? AllowAlias() : NeverAlias()
  return dag(aliasstyle, T)
end

dag(::Nothing) = nothing

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
