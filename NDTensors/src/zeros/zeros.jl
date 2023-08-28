struct Zeros{ElT,N,Axes,Alloc<:AbstractArray{ElT,N}} <: AbstractArray{ElT,N}
  z::FillArrays.Zeros{ElT,N,Axes}
  function NDTensors.Zeros{ElT,N,Alloc}(inds::Tuple) where {ElT,N,Alloc}
    z = FillArrays.Zeros{ElT,N}(inds)
    Axes = typeof(FillArrays.axes(z))
    return new{ElT,N,Axes,Alloc}(z)
  end
  function NDTensors.Zeros{ElT,N,Alloc}(::Tuple{}) where {ElT,N,Alloc}
    @assert N == 1
    z = FillArrays.Zeros{ElT,N}(1)
    Axes = typeof(FillArrays.axes(z))
    return new{ElT,N,Axes,Alloc}(z)
  end

  function NDTensors.Zeros{ElT,N,Axes,Alloc}(inds::Tuple) where {ElT,N,Axes,Alloc}
    @assert Axes == typeof(Base.axes(inds))
    z = FillArrays.Zeros{ElT,N}(inds)
    return new{ElT,N,Axes,Alloc}(z)
  end
end

function Zeros(alloc::Type{<:AbstractArray}, inds...)
  @assert ndims(alloc) == length(inds...)
  return Zeros{eltype(alloc),ndims(alloc),alloc}(Tuple(inds))
end

function Zeros{ElT}(alloc::Type{<:AbstractArray}, inds...) where {ElT}
  alloc = set_eltype(alloc, ElT)
  return Zeros(alloc, inds)
end

Base.ndims(::NDTensors.Zeros{ElT,N}) where {ElT,N} = N
ndims(::NDTensors.Zeros{ElT,N}) where {ElT,N} = N
Base.eltype(::Zeros{ElT}) where {ElT} = ElT
alloctype(::NDTensors.Zeros{ElT,N,Axes,Alloc}) where {ElT,N,Axes,Alloc} = Alloc
alloctype(::Type{<:NDTensors.Zeros{ElT,N,Axes,Alloc}}) where {ElT,N,Axes,Alloc} = Alloc
axes(::Type{<:NDTensors.Zeros{ElT,N,Axes}}) where {ElT,N,Axes} = Axes

Base.size(zero::Zeros) = Base.size(zero.z)

Base.print_array(io::IO, X::Zeros) = Base.print_array(io, X.z)

data(zero::Zeros) = zero.z
getindex(zero::Zeros) = getindex(zero.z)

array(zero::Zeros) = alloctype(zero)(zero.z)
Array(zero::Zeros) = array(zero)
axes(z::NDTensors.Zeros) = axes(z.z)
dims(z::Zeros) = Tuple(size(z.z))
dim(z::Zeros) = size(z.z)
copy(z::Zeros) = Zeros{eltype(z),1,alloctype(z)}(dims(z))

Base.convert(x::Type{T}, z::NDTensors.Zeros) where {T<:Array} = Base.convert(x, z.z)

function complex(z::Zeros)
  ElT = complex(eltype(z))
  N = ndims(z)
  AllocT = similartype(alloctype(z), ElT)
  return NDTensors.Zeros{ElT,N,AllocT}(dims(z))
end

Base.getindex(a::Zeros, i) = Base.getindex(a.z, i)
Base.sum(z::Zeros) = sum(z.z)
LinearAlgebra.norm(z::Zeros) = norm(z.z)
setindex!(A::NDTensors.Zeros, v, I) = setindex!(A.z, v, I)

function (arraytype::Type{<:Zeros})(::AllowAlias, A::Zeros)
  return A
end

function (arraytype::Type{<:Zeros})(::NeverAlias, A::Zeros)
  return copy(A)
end

function to_shape(::Type{<:Zeros}, dims::Tuple)
  return NDTensors.to_shape(dims)
end

function promote_rule(z1::Type{<:Zeros}, z2::Type{<:Zeros})
  ElT = promote_type(eltype(z1), eltype(z2))
  @assert ndims(z1) == ndims(z2)
  Axs = axes(z1)
  Alloc = promote_type(alloctype(z1), alloctype(z2))
  set_eltype(Alloc, ElT)
  return NDTensors.Zeros{ElT,ndims(z1),Axs,Alloc}
end

function promote_rule(z1::Type{<:Zeros}, z2::Type{<:AbstractArray})
  ElT = promote_type(eltype(z1), eltype(z2))
  @assert ndims(z1) == ndims(z2)
  Axs = axes(z1)
  Alloc = promote_type(alloctype(z1), z2)
  set_eltype(Alloc, ElT)
  return NDTensors.Zeros{ElT,ndims(z1),Axs,Alloc}
end

function promote_rule(z1::Type{<:AbstractArray}, z2::Type{<:Zeros})
  return promote_rule(z2, z1)
end

## Check datatypes to see if underlying storage is a 
## NDTensors.Zeros
is_unallocated_zeros(a) = data_isa(a, NDTensors.Zeros)

function allocate(T::Tensor)
  if !is_unallocated_zeros(T)
    return T
  end
  type = similartype(T)
  alloc_data = allocate(data(T))
  is = inds(T)
  @show typeof(alloc_data)
  return type(allocate(data(T)), inds(T))
end

function allocate(T::Tensor, elt::Type)
  if !is_unallocated_zeros(T)
    return T
  end
  ## allocate the tensor if is_unallocated_zeros
  ElT = promote_type(eltype(data(T)), elt)
  @show ElT
  d = similartype(alloctype(data(T)), ElT)(undef, dim(to_shape(typeof(data(T)), inds(T))))
  fill!(d, 0)
  return Tensor(d, inds(T))
end

allocate(d::AbstractArray) = d

function allocate(z::Zeros)
  return alloctype(z)(undef, dims(z))
end
