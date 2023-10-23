## TODO still working to make this implementation simplified
struct UnallocatedZeros{ElT,N,Axes,Alloc<:AbstractArray{ElT,N}} <:
       FillArrays.AbstractZeros{ElT,N,Axes}
  z::FillArrays.Zeros{ElT,N,Axes}

  function UnallocatedZeros{ElT,N,Axes,Alloc}(inds::Tuple) where {ElT,N,Axes,Alloc}
    z = FillArrays.Zeros(inds)
    ax = typeof(FillArrays.axes(z))
    return new{ElT,N,ax,Alloc}(z)
  end
end

data(Z::UnallocatedZeros) = Z.z
copy(Z::UnallocatedZeros) = typeof(Z)(size(Z))
Base.vec(Z::UnallocatedZeros) = typeof(Z)(length(Z))

function complex(z::UnallocatedZeros)
  ElT = complex(eltype(z))
  N = ndims(z)
  AllocT = similartype(alloctype(z), ElT)
  return UnallocatedZeros{ElT,N,AllocT}(dims(z))
end

## Check datatypes to see if underlying storage is a 
## UnallocatedZeros
#is_unallocated_zeros(a) = data_isa(a, UnallocatedZeros)

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
