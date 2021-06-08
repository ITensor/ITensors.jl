# ITensor Index identity: dimension labels and Einstein notation

Many tensor contraction libraries use [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation),
such as [NumPy's einsum function](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), [ncon](https://arxiv.org/abs/1402.0939), and various Julia packages such as [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl), [Tullio.jl](https://github.com/mcabbott/Tullio.jl), [OMEinsum.jl](https://github.com/under-Peter/OMEinsum.jl), and [Einsum.jl](https://github.com/ahwillia/Einsum.jl), among others.

ITensor also uses Einstein notation, however the labels are stored inside the tensor and carried around with them during various operations. In addition, the labels that determine if tensor indices match with each other, and therefore automatically contract when doing `*` or match when adding or subtracting, are more sophisticated than simple characters or strings. ITensor indices are given a unique random ID number when they are constructed, and additionally users can add additional information like prime levels and tags which uniquely determine an Index. This is in contrast to simpler implementations of the same idea, such as the [NamedDims.jl](https://github.com/invenia/NamedDims.jl) package, which only allow symbols as the metadata for uniquely identifying a tensor/array dimension.

```@setup itensor
using ITensors
using Random
Random.seed!(1)
```

## Index identity

Here is an illustration of how the different types of Index metadata (random ID, prime level, and tags) work for Index identity:
```@repl itensor
i = Index(2)
j = Index(2)
i == j
id(i)
id(j)
ip = i'
ip == i
plev(i) == 0
plev(ip) == 1
noprime(ip) == i
ix = addtags(i, "x")
ix == i
removetags(ix, "x") == i
ixyz = addtags(ix, "y,z")
ixyz == addtags(i, "z,y,x")
```

The different metadata that are stored inside of ITensor indices that determine their identity are useful in different contexts. The random ID is particularly useful in the case when a new Index needs to be generated internally by ITensor, such as when performing a matrix factorization. In the case of a matrix factorization, we want to make sure that the new Index will not accidentally clash with an existing one, for example:
```@repl itensor
i = Index(2, "i")
j = Index(2, "j")
A = randomITensor(i, j)
U, S, V = svd(A, i; lefttags="i", righttags="j");
inds(U)
inds(S)
inds(V)
norm(U * S * V - A)
```
You can see that it would have been a problem here if there wasn't a new ID assigned to the Index, since it would have clashed with the original index. In this case, it could be avoided by giving the new indices different tags (with the keyword arguments `lefttags` and `righttags`), but in more complicated examples where it is not practical to do that (such as a case where many new indices are being introduced, for example for a tensor train (TT)/matrix product state (MPS)), it is convenient to not force users to come up with unique prime levels or tags themselves. It can also help to avoid accidental contractions in more complicated tensor network algorithms where there are many indices that can potentially have the same prime levels or tags.

In contrast, using multiple indices with the same Index ID but different prime levels and tags can be useful in situations where there is a more fundamental relationship between the spaces. For example, in the case of an ITensor corresponding to a Hermitian operator, it is helpful to make the bra space and ket spaces the same up to a prime level:
```repl itensor
i = Index(2, "i")
j = Index(3, "j")
A = randomITensor(i', j', dag(i), dag(j))
H = 0.5 * (A + swapprime(dag(A), 0 => 1))
v = randomITensor(i, j)
Hv = noprime(H * v)
vH = dag(v)' * H
norm(Hv - dag(vH))
```
Note that we have added `dag` in a few places, which is superfluous in this case since the tensors are real and dense but become important when the tensors are complex and/or have symmetries.
You can see that in this case, it is very useful to relate the bra and ket spaces by prime levels, since it makes it much easier to perform operations that map from one space to another. We could have created `A` from 4 entirely different indices with different ID numbers, but it would make the operations a bit more cumbersome, as shown below:
```@repl itensor
i = Index(2, "i")
j = Index(3, "j")
ip = Index(2, "i")
jp = Index(3, "jp")
A = randomITensor(ip, jp, dag(i), dag(j))
H = 0.5 * (A + swapinds(dag(A), (i, j), (ip, jp)))
v = randomITensor(i, j)
Hv = replaceinds(H * v, (ip, jp) => (i, j))
vH = replaceinds(dag(v), (i, j) => (ip, jp)) * H
norm(Hv - dag(vH))
```

## Relationship to other Einstein notation-based libraries

Here we show examples of different ways to perform the contraction
`"ab,bc,cd->ad"` in ITensor.

```@repl itensor
da, dc = 2, 3;
db, dd = da, dc;
tags = ("a", "b", "c", "d");
dims = (da, db, dc, dd);
a, b, c, d = Index.(dims, tags);
Aab = randomITensor(a, b)
Bbc = randomITensor(b, c)
Ccd = randomITensor(c, d)

# "ab,bc,cd->ad"
out1 = Aab * Bbc * Ccd
@show hassameinds(out1, (a, d))

#
# Using replaceinds (most general way)
#

# "ba,bc,dc->ad"
Aba = replaceinds(Aab, (a, b) => (b, a))
Cdc = replaceinds(Ccd, (c, d) => (d, c))
out2 = Aba * Bbc * Cdc
@show hassameinds(out2, (a, d))

#
# Using setinds
#

# This is a bit lower level
# since it doesn't check if the indices
# are compatible in dimension,
# so is not recommended in general.
using ITensors: setinds

Aba = setinds(Aab, (b, a))
Cdc = setinds(Ccd, (d, c))
out2 = Aba * Bbc * Cdc
@show hassameinds(out2, (a, d))

#
# Using prime levels (assuming
# the indices were made with these
# prime levels in the first place)
#

a = Index(da, "a")
c = Index(dc, "c")
b, d = a', c'
Aab = randomITensor(a, b)
Bbc = randomITensor(b, c)
Ccd = randomITensor(c, d)
out1 = Aab * Bbc * Ccd
@show hassameinds(out1, (a, d))

Aba = swapprime(Aab, 0 => 1)
Cdc = swapprime(Ccd, 0 => 1)
out2 = Aba * Bbc * Cdc
@show hassameinds(out2, (a, d))

#
# Using tags (assuming
# the indices were made with these
# tags in the first place)
#

a = Index(da, "a")
c = Index(dc, "c")
b, d = settags(a, "b"), settags(c, "d")
Aab = randomITensor(a, b)
Bbc = randomITensor(b, c)
Ccd = randomITensor(c, d)
out1 = Aab * Bbc * Ccd
@show hassameinds(out1, (a, d))

Aba = swaptags(Aab, "a", "b")
Cdc = swaptags(Ccd, "c", "d")
out2 = Aba * Bbc * Cdc
@show hassameinds(out2, (a, d))

#
# Using Julia Arrays
#

A = randn(da, db)
B = randn(db, dc)
C = randn(dc, dd)

tags = ("a", "b", "c", "d")
dims = (da, db, dc, dd)
a, b, c, d = Index.(dims, tags)

Aab = itensor(A, a, b)
Bbc = itensor(B, b, c)
Ccd = itensor(C, c, d)
out1 = Aab * Bbc * Ccd
@show hassameinds(out1, (a, d))

Aba = itensor(A, b, a)
Cdc = itensor(C, d, c)
out2 = Aba * Bbc * Cdc
@show hassameinds(out2, (a, d))

#
# Note that we may start allowing
# this notation in future:
# (https://github.com/ITensor/ITensors.jl/issues/673)
#
#out1 = A[a, b] * B[b, c] * C[c, d]
#@show hassameinds(out1, (a, d))
#
#out2 = A[b, a] * B[b, c] * C[d, c]
#@show hassameinds(out2, (a, d))
```

