struct UnallocatedZeros{ElT,N,Axes,Alloc<:AbstractArray{ElT,N}} <: AbstractArray{ElT,N}
  z::FillArrays.Zeros{ElT,N,Axes}
  function NDTensors.UnallocatedZeros{ElT,N,Alloc}(inds::Tuple) where {ElT,N,Alloc}
    z = FillArrays.Zeros{ElT,N}(inds)
    Axes = typeof(FillArrays.axes(z))
    return new{ElT,N,Axes,Alloc}(z)
  end
  function NDTensors.UnallocatedZeros{ElT,N,Alloc}(::Tuple{}) where {ElT,N,Alloc}
    @assert N == 1
    z = FillArrays.Zeros{ElT,N}(1)
    Axes = typeof(FillArrays.axes(z))
    return new{ElT,N,Axes,Alloc}(z)
  end

  function NDTensors.UnallocatedZeros{ElT,N,Axes,Alloc}(
    inds::Tuple
  ) where {ElT,N,Axes,Alloc}
    @assert Axes == typeof(Base.axes(inds))
    z = FillArrays.Zeros{ElT,N}(inds)
    return new{ElT,N,Axes,Alloc}(z)
  end
end

function UnallocatedZeros(alloc::Type{<:AbstractArray}, inds...)
  @assert ndims(alloc) == length(inds...)
  return UnallocatedZeros{eltype(alloc),ndims(alloc),alloc}(Tuple(inds))
end

function UnallocatedZeros{ElT}(alloc::Type{<:AbstractArray}, inds...) where {ElT}
  alloc = set_eltype(alloc, ElT)
  return UnallocatedZeros(alloc, inds)
end

Base.ndims(::NDTensors.UnallocatedZeros{ElT,N}) where {ElT,N} = N
ndims(::NDTensors.UnallocatedZeros{ElT,N}) where {ElT,N} = N
Base.eltype(::UnallocatedZeros{ElT}) where {ElT} = ElT
alloctype(::NDTensors.UnallocatedZeros{ElT,N,Axes,Alloc}) where {ElT,N,Axes,Alloc} = Alloc
function alloctype(
  ::Type{<:NDTensors.UnallocatedZeros{ElT,N,Axes,Alloc}}
) where {ElT,N,Axes,Alloc}
  return Alloc
end
axes(::Type{<:NDTensors.UnallocatedZeros{ElT,N,Axes}}) where {ElT,N,Axes} = Axes

Base.size(zero::UnallocatedZeros) = Base.size(zero.z)

Base.print_array(io::IO, X::UnallocatedZeros) = Base.print_array(io, X.z)

data(zero::UnallocatedZeros) = zero.z
getindex(zero::UnallocatedZeros) = getindex(zero.z)

array(zero::UnallocatedZeros) = alloctype(zero)(zero.z)
Array(zero::UnallocatedZeros) = array(zero)
axes(z::NDTensors.UnallocatedZeros) = axes(z.z)
dims(z::UnallocatedZeros) = Tuple(size(z.z))
dim(z::UnallocatedZeros) = size(z.z)
copy(z::UnallocatedZeros) = UnallocatedZeros{eltype(z),1,alloctype(z)}(dims(z))

function Base.convert(x::Type{T}, z::NDTensors.UnallocatedZeros) where {T<:Array}
  return Base.convert(x, z.z)
end

function complex(z::UnallocatedZeros)
  ElT = complex(eltype(z))
  N = ndims(z)
  AllocT = similartype(alloctype(z), ElT)
  return NDTensors.UnallocatedZeros{ElT,N,AllocT}(dims(z))
end

Base.getindex(a::UnallocatedZeros, i) = Base.getindex(a.z, i)
Base.sum(z::UnallocatedZeros) = sum(z.z)
LinearAlgebra.norm(z::UnallocatedZeros) = norm(z.z)
setindex!(A::NDTensors.UnallocatedZeros, v, I) = setindex!(A.z, v, I)

function (arraytype::Type{<:UnallocatedZeros})(::AllowAlias, A::UnallocatedZeros)
  return A
end

function (arraytype::Type{<:UnallocatedZeros})(::NeverAlias, A::UnallocatedZeros)
  return copy(A)
end

function to_shape(::Type{<:UnallocatedZeros}, dims::Tuple)
  return NDTensors.to_shape(dims)
end

function promote_rule(z1::Type{<:UnallocatedZeros}, z2::Type{<:UnallocatedZeros})
  ElT = promote_type(eltype(z1), eltype(z2))
  @assert ndims(z1) == ndims(z2)
  Axs = axes(z1)
  Alloc = promote_type(alloctype(z1), alloctype(z2))
  set_eltype(Alloc, ElT)
  return NDTensors.UnallocatedZeros{ElT,ndims(z1),Axs,Alloc}
end

function promote_rule(z1::Type{<:UnallocatedZeros}, z2::Type{<:AbstractArray})
  ElT = promote_type(eltype(z1), eltype(z2))
  @assert ndims(z1) == ndims(z2)
  Axs = axes(z1)
  Alloc = promote_type(alloctype(z1), z2)
  set_eltype(Alloc, ElT)
  return NDTensors.UnallocatedZeros{ElT,ndims(z1),Axs,Alloc}
end

function promote_rule(z1::Type{<:AbstractArray}, z2::Type{<:UnallocatedZeros})
  return promote_rule(z2, z1)
end

## Check datatypes to see if underlying storage is a 
## NDTensors.UnallocatedZeros
is_unallocated_zeros(a) = data_isa(a, NDTensors.UnallocatedZeros)

function allocate(T::Tensor)
  if !is_unallocated_zeros(T)
    return T
  end
  return tensor(set_datatype(typeof(NDTensors.storage(T)), alloctype(data(T)))(allocate(data(T))), inds(T))
  #@show convert(type, out_data)
  #return type(allocate(data(T)), inds(T))
end

function allocate(T::Tensor, elt::Type)
  if !is_unallocated_zeros(T)
    return T
  end
  ## allocate the tensor if is_unallocated_zeros
  ElT = promote_type(eltype(data(T)), elt)
  d = similartype(alloctype(data(T)), ElT)(undef, dim(to_shape(typeof(data(T)), inds(T))))
  fill!(d, 0)
  return Tensor(d, inds(T))
end

allocate(d::AbstractArray) = d

function allocate(z::UnallocatedZeros)
  return alloctype(z)(undef, dims(z))
end
