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
