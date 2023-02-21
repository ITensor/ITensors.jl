export dense, dims, dim, mindim, diaglength

# dim and dims are used in the Tensor interface, overload 
# base Dims here
dims(ds::Dims) = ds

# Generic dims function
dims(inds::Tuple) = ntuple(i -> dim(@inbounds inds[i]), Val(length(inds)))

# Generic dim function
dim(inds::Tuple) = prod(dims(inds))

dims(::Tuple{}) = ()

dim(::Tuple{}) = 1

dense(ds::Dims) = ds

dense(::Type{DimsT}) where {DimsT<:Dims} = DimsT

dim(ds::Dims) = prod(ds)

dim(ds::Dims, i::Int) = dims(ds)[i]

mindim(inds::Tuple) = minimum(dims(inds))

mindim(::Tuple{}) = 1

diaglength(inds::Tuple) = mindim(inds)

"""
    dim_to_strides(ds)

Get the strides from the dimensions.

This is unexported, call with NDTensors.dim_to_strides.
"""
dim_to_strides(ds) = Base.size_to_strides(1, dims(ds)...)

"""
    dim_to_stride(ds, k::Int)

Get the stride of the dimension k from the dimensions.

This is unexported, call with NDTensors.stride.
"""
dim_to_stride(ds, k::Int) = dim_to_strides(ds)[k]

# This is to help with some generic programming in the Tensor
# code (it helps to construct a Tuple(::NTuple{N,Int}) where the 
# only known thing for dispatch is a concrete type such
# as Dims{4})
similartype(::Type{<:Dims}, ::Type{Val{N}}) where {N} = Dims{N}

# This is to help with ITensor compatibility
dim(i::Int) = i

# This is to help with ITensor compatibility
dir(::Int) = 0

# This is to help with ITensor compatibility
dag(i::Int) = i

# This is to help with ITensor compatibility
sim(i::Int) = i

#
# Order value type
#

# More complicated definition makes Order(Ref(2)[]) faster
@eval struct Order{N}
  (OrderT::Type{<:Order})() = $(Expr(:new, :OrderT))
end

@doc """
    Order{N}

A value type representing the order of an ITensor.
""" Order

"""
    Order(N) = Order{N}()

Create an instance of the value type Order representing
the order of an ITensor.
"""
Order(N) = Order{N}()

#dims for tensor
# The size is obtained from the indices
dims(T::Tensor) = dims(inds(T))
dim(T::Tensor) = dim(inds(T))
dim(T::Tensor, i::Int) = dim(inds(T), i)
maxdim(T::Tensor) = maxdim(inds(T))
mindim(T::Tensor) = mindim(inds(T))
diaglength(T::Tensor) = mindim(T)
