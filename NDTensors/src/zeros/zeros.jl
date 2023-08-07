struct Zeros{ElT,N,Axes,Alloc<:AbstractArray} <: AbstractArray{ElT,N}
  z::FillArrays.Zeros{ElT,N,Axes}
  function NDTensors.Zeros{ElT,N,Alloc}(inds::Tuple) where {ElT,N,Alloc}
    @assert eltype(Alloc) == ElT
    z = FillArrays.Zeros{ElT,N}(inds)
    Axes = axes(z)
    return new{ElT,N,Axes,Alloc}(z)
  end
end

Base.ndims(::NDTensors.Zeros{ElT,N}) where {ElT,N} = N
ndims(::NDTensors.Zeros{ElT,N}) where {ElT,N} = N
Base.eltype(::Zeros{ElT}) where {ElT} = ElT
datatype(::NDTensors.Zeros{ElT,N,DataT}) where {ElT,N,DataT} = DataT
datatype(::Type{<:NDTensors.Zeros{ElT,N,DataT}}) where {ElT,N,DataT} = DataT

Base.size(zero::Zeros) = Base.size(zero.z)

Base.print_array(io::IO, X::Zeros) = Base.print_array(io, X.z)

data(zero::Zeros) = zero.z
getindex(zero::Zeros) = getindex(zero.z)

array(zero::Zeros) = datatype(zero)(zero.z)
Array(zero::Zeros) = array(zero)
dims(z::Zeros) = axes(z.z)
copy(z::Zeros) = Zeros{eltype(z),1,datatype(z)}(dims(z))

Base.convert(x::Type{T}, z::NDTensors.Zeros) where {T<:Array} = Base.convert(x, z.z)

Base.getindex(a::Zeros, i) = Base.getindex(a.z, i)
Base.sum(z::Zeros) = sum(z.z)
LinearAlgebra.norm(z::Zeros) = norm(z.z)
setindex!(A::NDTensors.Zeros, v, I) = setindex!(A.z, v, I)

is_unallocated_zeros(t::Tensor) = is_unallocated_zeros(storage(t))
is_unallocated_zeros(st::TensorStorage) = data(st) isa Zeros

function (arraytype::Type{<:Zeros})(::AllowAlias, A::Zeros)
  return A
end

function (arraytype::Type{<:Zeros})(::NeverAlias, A::Zeros)
  return copy(A)
end
