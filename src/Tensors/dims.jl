export dense,
       dims,
       dim,
       dir,
       dag

#
# Tools for working with Dims/Tuples
# TODO: put this in a seperate file
#

# dim and dims are used in the Tensor interface, overload 
# base Dims here
dims(ds::Dims) = ds
dims(::Tuple{}) = ()
dense(ds::Dims) = ds
dense(::Type{DimsT}) where {DimsT<:Dims} = DimsT
dim(ds::Dims) = prod(ds)
dim(ds::Dims,i::Int) = dims(ds)[i]

Base.ndims(::Dims{N}) where {N} = N
Base.ndims(::Type{Dims{N}}) where {N} = N
Base.ndims(::Tuple{}) = 0
Base.ndims(::Type{Tuple{}}) = 0

mindim(inds) = (ndims(inds) == 0 ? 1 : minimum(dims(inds)))
diaglength(inds) = mindim(inds)

# This may be a bad idea to overload?
# Type piracy?
Base.strides(ds::Dims) = Base.size_to_strides(1, dims(ds)...)
Base.copy(ds::Dims) = ds

## TODO: should this be StaticArrays.similar_type?
#Base.promote_rule(::Type{<:Dims},
#                  ::Type{Val{N}}) where {N} = Dims{N}

# This is to help with some generic programming in the Tensor
# code (it helps to construct a Tuple(::NTuple{N,Int}) where the 
# only known thing for dispatch is a concrete type such
# as Dims{4})
StaticArrays.similar_type(::Type{<:Dims},
                          ::Type{Val{N}}) where {N} = Dims{N}

# This is to help with ITensor compatibility
dir(::Int) = 0

dag(i::Int) = i

