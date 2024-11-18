# SparseArraysBase

SparseArraysBase is a package that aims to expand on the sparse array functionality that is currently in Julia Base.
While SparseArrays.jl is centered mostly around `SparseMatrixCSC` and the SuiteSparse library, here we wish to broaden the scope a bit, and consider generic sparse arrays.
Abstractly, the mental model can be considered as a storage object that holds the stored values, and a bijection between the array indices and the indices of the storage.
For now, we focus on providing efficient implementations of Dictionary of Key (DOK) type sparse storage formats, but may expand upon this in the future.
As a result, for typical linear algebra routines, we still expect `SparseMatrixCSC` to be the object of choice.

The design consists of roughly three components:
- `AbstractSparseArray` interface functions
- Overloaded Julia base methods
- `SparseArrayDOK` struct that implements this

## AbstractSparseArray

The first part consists of typical functions that are useful in the context of sparse arrays.
The minimal interface, which enables the usage of the rest of this package, consists of the following functions:

| Signature | Description | Default |
|-----------|-------------|---------|
| `sparse_storage(a::AbstractArray)` | Returns the storage object of the sparse array | `a` |
| `storage_index_to_index(a::AbstractArray, I)` | Converts a storage index to an array index | `I` |
| `index_to_storage_index(a::AbstractArray, I)` | Converts an array index to a storage index | `I` |

Using these primitives, several convenience functions are defined to facilitate the writing of sparse array algorithms.

| Signature | Description | Default |
|-----------|-------------|---------|
| `storage_indices(a)` | Returns the indices of the storage | `eachindex(sparse_storage(a))` |
| `stored_indices(a)` | Returns the indices of the stored values | `Iterators.map(Base.Fix1(storage_index_to_index, a), storage_indices(a))` |
| `stored_length(a)` | Returns the number of stored values | `length(storage_indices(a))` |

<!-- TODO: `getindex!`, `increaseindex!`, `sparse_map`, expose "zero" functionality?  -->

Interesting to note here is that the design is such that we can define sparse arrays without having to subtype `AbstractSparseArray`.
To achieve this, each function `f` is defined in terms of `sparse_f`, rather than directly overloading `f`.
<!--
TODO:
In order to opt-in to the sparse array functionality, one needs to dispatch the functions through `sparse_f` instead of `f`.
For convenience, you can automatically dispatch all functions through `sparse_f` by using the following macro:

```julia
@abstractsparsearray MySparseArrayType
```
-->

## Overloaded Julia base methods

The second part consists of overloading Julia base methods to work with sparse arrays.
In particular, specialised implementations exist for the following functions:

- `sparse_similar`
- `sparse_reduce`
- `sparse_map`
- `sparse_map!`
- `sparse_all`
- `sparse_any`
- `sparse_isequal`
- `sparse_fill!`
- `sparse_zero`, `sparse_zero!`, `sparse_iszero`
- `sparse_one`, `sparse_one!`, `sparse_isone`
- `sparse_reshape`, `sparse_reshape!`
- `sparse_cat`, `sparse_cat!`
- `sparse_copy!`, `sparse_copyto!`
- `sparse_permutedims`, `sparse_permutedims!`
- `sparse_mul!`, `sparse_dot`

## SparseArrayDOK

Finally, the `SparseArrayDOK` struct is provided as a concrete implementation of the `AbstractSparseArray` interface.
It is a dictionary of keys (DOK) type sparse array, which stores the values in a `Dictionaries.jl` dictionary, and maps the indices to the keys of the dictionary.
This model is particularly useful for sparse arrays with a small number of non-zero elements, or for arrays that are constructed incrementally, as it boasts fast random accesses and insertions.
The drawback is that sequential iteration is slower than for other sparse array types, leading to slower linear algebra operations.
For the purposes of `SparseArraysBase`, this struct will serve as the canonical example of a sparse array, and will be returned by default when new sparse arrays are created.

One particular feature of `SparseArrayDOK` is that it can be used in cases where the non-stored entries have to be constructed in a non-trivial way.
Typically, sparse arrays use `zero(eltype(a))` to construct the non-stored entries, but this is not always sufficient.
A concrete example is found in `BlockSparseArrays.jl`, where initialization of the non-stored entries requires the construction of a block of zeros of appropriate size.

<!-- TODO: update TODOs -->

## TODO
Still need to implement `Base` functions:
```julia
[x] sparse_zero(a::AbstractArray) = similar(a)
[x] sparse_iszero(a::AbstractArray) = iszero(nonzero_length(a)) # Uses `all`, make `sparse_all`?
[x] sparse_one(a::AbstractArray) = ...
[x] sparse_isreal(a::AbstractArray) = ... # Uses `all`, make `sparse_all`?
[x] sparse_isequal(a1::AbstractArray, a2::AbstractArray) = ...
[x] sparse_conj!(a::AbstractArray) = conj!(nonzeros(a))
[x] sparse_reshape(a::AbstractArray, dims) = ...
[ ] sparse_all(f, a::AbstractArray) = ...
[ ] sparse_getindex(a::AbstractArray, 1:2, 2:3) = ... # Slicing
```
`LinearAlgebra` functions:
```julia
[ ] sparse_mul!
[ ] sparse_lmul!
[ ] sparse_ldiv!
[ ] sparse_rdiv!
[ ] sparse_axpby!
[ ] sparse_axpy!
[ ] sparse_norm
[ ] sparse_dot/sparse_inner
[ ] sparse_adoint!
[ ] sparse_transpose!

# Using conversion to `SparseMatrixCSC`:
[ ] sparse_qr
[ ] sparse_eigen
[ ] sparse_svd
```
`TensorAlgebra` functions:
```julia
[ ] add!
[ ] contract!
```
