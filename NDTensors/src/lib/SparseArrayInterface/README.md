# SparseArrayInterface

Defines a generic interface for sparse arrays in Julia.

The minimal interface is:
```julia
nonzeros(a::AbstractArray) = ...
nonzero_index_to_index(a::AbstractArray, Inz) = ...
index_to_nonzero_index(a::AbstractArray{<:Any,N}, I::CartesianIndex{N}) where {N} = ...
Broadcast.BroadcastStyle(arraytype::Type{<:AbstractArray}) = SparseArrayInterface.SparseArrayStyle{ndims(arraytype)}()
```
Once these are defined, along with Julia AbstractArray interface functions like
`Base.size` and `Base.similar`, functions like the following will take advantage of sparsity:
```julia
SparseArrayInterface.nonzero_length # SparseArrays.nnz
SparseArrayInterface.sparse_getindex
SparseArrayInterface.sparse_setindex!
SparseArrayInterface.sparse_map!
SparseArrayInterface.sparse_copy!
SparseArrayInterface.sparse_copyto!
SparseArrayInterface.sparse_permutedims!
```
which can be used to define the corresponding `Base` functions.

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
