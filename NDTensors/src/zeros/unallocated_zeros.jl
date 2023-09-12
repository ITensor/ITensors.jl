struct UnallocatedZeros{ElT,N,Axes,Alloc<:AbstractArray{ElT,N}} <:
       FillArrays.AbstractFill{ElT,N,Axes}
  z::FillArrays.Zeros{ElT,N,Axes}
  function UnallocatedZeros{ElT,N,Alloc}(inds::Tuple) where {ElT,N,Alloc}
    z = FillArrays.Zeros{ElT,N}(inds)
    Axes = typeof(FillArrays.axes(z))
    return new{ElT,N,Axes,Alloc}(z)
  end
  function UnallocatedZeros{ElT,N,Alloc}(::Tuple{}) where {ElT,N,Alloc}
    @assert N == 1
    z = FillArrays.Zeros{ElT,N}(1)
    Axes = typeof(FillArrays.axes(z))
    return new{ElT,N,Axes,Alloc}(z)
  end

  function UnallocatedZeros{ElT,N,Axes,Alloc}(inds::Tuple) where {ElT,N,Axes,Alloc}
    @assert Axes == typeof(Base.axes(inds))
    z = FillArrays.Zeros{ElT,N}(inds)
    return new{ElT,N,Axes,Alloc}(z)
  end
end

function UnallocatedZeros{ElT,N,Axes,Alloc}() where {ElT,N,Axes,Alloc<:AbstractArray{ElT,N}}
  return UnallocatedZeros{ElT,N,Axes,Alloc}[]
end
function UnallocatedZeros(alloc::Type{<:AbstractArray}, inds...)
  @assert ndims(alloc) == length(inds...)
  return UnallocatedZeros{eltype(alloc),ndims(alloc),alloc}(Tuple(inds))
end

function UnallocatedZeros{ElT}(alloc::Type{<:AbstractArray}, inds...) where {ElT}
  alloc = set_eltype(alloc, ElT)
  return UnallocatedZeros(alloc, inds)
end

Base.ndims(::UnallocatedZeros{ElT,N}) where {ElT,N} = N
ndims(::UnallocatedZeros{ElT,N}) where {ElT,N} = N
Base.eltype(::UnallocatedZeros{ElT}) where {ElT} = ElT
alloctype(::UnallocatedZeros{ElT,N,Axes,Alloc}) where {ElT,N,Axes,Alloc} = Alloc
function alloctype(::Type{<:UnallocatedZeros{ElT,N,Axes,Alloc}}) where {ElT,N,Axes,Alloc}
  return Alloc
end
axes(::Type{<:UnallocatedZeros{ElT,N,Axes}}) where {ElT,N,Axes} = Axes

Base.size(zero::UnallocatedZeros) = Base.size(zero.z)

Base.print_array(io::IO, X::UnallocatedZeros) = Base.print_array(io, X.z)

data(zero::UnallocatedZeros) = zero.z
getindex(zero::UnallocatedZeros) = getindex(zero.z)

array(zero::UnallocatedZeros) = alloctype(zero)(zero.z)
Array(zero::UnallocatedZeros) = array(zero)
axes(z::UnallocatedZeros) = axes(z.z)
dims(z::UnallocatedZeros) = Tuple(size(z.z))
dim(z::UnallocatedZeros) = size(z.z)
copy(z::UnallocatedZeros) = UnallocatedZeros{eltype(z),1,alloctype(z)}(dims(z))
Base.vec(z::Type{<:UnallocatedZeros}) = z
Base.vec(z::UnallocatedZeros) = z
function Base.convert(x::Type{T}, z::UnallocatedZeros) where {T<:Array}
  return Base.convert(x, z.z)
end

function complex(z::UnallocatedZeros)
  ElT = complex(eltype(z))
  N = ndims(z)
  AllocT = similartype(alloctype(z), ElT)
  return UnallocatedZeros{ElT,N,AllocT}(dims(z))
end

Base.sum(z::UnallocatedZeros) = sum(z.z)
LinearAlgebra.norm(z::UnallocatedZeros) = norm(z.z)

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
  return UnallocatedZeros{ElT,ndims(z1),Axs,Alloc}
end

function promote_rule(z1::Type{<:UnallocatedZeros}, z2::Type{<:AbstractArray})
  ElT = promote_type(eltype(z1), eltype(z2))
  @assert ndims(z1) == ndims(z2)
  Axs = axes(z1)
  Alloc = promote_type(alloctype(z1), z2)
  set_eltype(Alloc, ElT)
  return UnallocatedZeros{ElT,ndims(z1),Axs,Alloc}
end

function promote_rule(z1::Type{<:AbstractArray}, z2::Type{<:UnallocatedZeros})
  return promote_rule(z2, z1)
end

## Check datatypes to see if underlying storage is a 
## UnallocatedZeros
is_unallocated_zeros(a) = data_isa(a, UnallocatedZeros)

FillArrays.getindex_value(Z::UnallocatedZeros) = FillArrays.getindex_value(Z.z)

function generic_zeros(::Type{<:UnallocatedZeros}, inds::Integer)
  elt = default_eltype()
  datat = default_datatype(elt)
  N = ndims(datat)
  return UnallocatedZeros{elt,N,datat}(Tuple(dim))
end

function generic_zeros(::Type{<:UnallocatedZeros{ElT}}, inds::Integer) where {ElT}
  datat = default_datatype(ElT)
  N = ndims(datat)
  return UnallocatedZeros{ElT,N,datat}(Tuple(dim))
end

function generic_zeros(
  ::Type{<:UnallocatedZeros{ElT,N,DataT}}, dim::Integer
) where {ElT,N,DataT<:AbstractArray{ElT,N}}
  return UnallocatedZeros{ElT,N,DataT}(Tuple(dim))
end

function generic_zeros(
  ::Type{<:UnallocatedZeros{ElT,N,Axes,DataT}}, dim::Integer
) where {ElT,N,Axes,DataT<:AbstractArray{ElT,N}}
  return UnallocatedZeros{ElT,N,DataT}(Tuple(dim))
end
