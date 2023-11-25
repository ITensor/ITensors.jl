# # NamedDimsArrays.jl

using NDTensors.NamedDimsArrays: align, dimnames, named, unname
using NDTensors.TensorAlgebra: TensorAlgebra

## Named dimensions
i = named(2, "i")
j = named(2, "j")
k = named(2, "k")

## Arrays with named dimensions
na1 = randn(i, j)
na2 = randn(j, k)

@show dimnames(na1) == ("i", "j")

## Indexing
@show na1[j => 2, i => 1] == na1[1, 2]

## Tensor contraction
na_dest = TensorAlgebra.contract(na1, na2)

@show issetequal(dimnames(na_dest), ("i", "k"))
## `unname` removes the names and returns an `Array`
@show unname(na_dest, (i, k)) ≈ unname(na1) * unname(na2)

## Permute dimensions (like `ITensors.permute`)
na1 = align(na1, (j, i))
@show na1[i => 1, j => 2] == na1[2, 1]
