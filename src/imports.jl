import Adapt: adapt_storage, adapt_structure
import Base.Broadcast: # functions
    _broadcast_getindex, # types
    AbstractArrayStyle, BroadcastStyle, Broadcasted, DefaultArrayStyle, Style,
    broadcastable, broadcasted, instantiate
import Base: !, # functions
    adjoint, # macros
    @propagate_inbounds, # symbols
    +, # types
    Array, *, -, /, <, ==, >, CartesianIndices, NTuple, Tuple, Vector, ^, allunique, axes,
    complex, conj, convert, copy, copyto!, deepcopy, deleteat!, eachindex, eltype, fill!,
    filter, filter!, findall, findfirst, getindex, hash, imag, intersect, intersect!,
    isapprox, isassigned, isempty, isless, isreal, iszero, iterate, keys, lastindex, length,
    map, map!, ndims, print, promote_rule, push!, real, resize!, setdiff, setdiff!,
    setindex!, show, similar, size, summary, truncate, zero
import ITensors.NDTensors: # Deprecated
    addblock!, # Modules
    Strided, # to control threading
    # Types
    AliasStyle, AllowAlias, NeverAlias, array, blockdim, blockoffsets, contract, datatype,
    dense, denseblocks, diaglength, dim, dims, disable_tblis, eachnzblock, enable_tblis,
    ind, inds, insert_diag_blocks!, insertblock!, matrix, maxdim, mindim, nblocks, nnz,
    nnzblocks, nzblock, nzblocks, one, outer, permuteblocks, polar, ql, scale!, setblock!,
    setblockdim!, setinds, setstorage, sim, storage, storagetype, store, sum, tensor,
    truncate!, using_tblis, vector
import ITensors.Ops: Prod, Sum, terms
import LinearAlgebra: axpby!, axpy!, diag, dot, eigen, exp, factorize, ishermitian, lmul!,
    lq, mul!, norm, normalize, normalize!, nullspace, qr, rmul!, svd, tr, transpose
import Random: randn!
using ITensors.NDTensors.GPUArraysCoreExtensions: cpu
using ITensors.NDTensors: @Algorithm_str, Algorithm, EmptyNumber, _NTuple, _Tuple,
    blas_get_num_threads, disable_auto_fermion, double_precision, eachblock, eachdiagblock,
    enable_auto_fermion, fill!!, permutedims, permutedims!, randn!!, single_precision,
    timer, using_auto_fermion
using NDTensors.CUDAExtensions: cu
using SerializedElementArrays: SerializedElementVector

const DiskVector{T} = SerializedElementVector{T}

import SerializedElementArrays: disk
