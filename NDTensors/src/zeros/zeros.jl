struct Zeros{ElT,N,DataT} <: AbstractArray{ElT,N}
  z::FillArrays.Zeros
  is::Tuple
  function NDTensors.Zeros{ElT,N,DataT}(inds::Tuple) where {ElT,N,DataT}
    @assert eltype(DataT) == ElT
    @assert ndims(DataT) == N
    z = FillArrays.Zeros(ElT, dim(inds))
    return new{ElT,N,DataT}(z, inds)
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
dims(z::Zeros) = z.is
copy(z::Zeros) = Zeros{eltype(z),1,datatype(z)}(dims(z))

Base.convert(x::Type{T}, z::NDTensors.Zeros) where {T<:Array} = Base.convert(x, z.z)

Base.getindex(a::Zeros, i) = Base.getindex(a.z, i)
Base.sum(z::Zeros) = sum(z.z)
LinearAlgebra.norm(z::Zeros) = norm(z.z)
setindex!(A::NDTensors.Zeros, v, I) = setindex!(A.z, v, I)

Base.iszero(t::Tensor) = iszero(storage(t))
Base.iszero(st::TensorStorage) = data(st) isa Zeros

function (arraytype::Type{<:Zeros})(::AllowAlias, A::Zeros)
  return A
end

function (arraytype::Type{<:Zeros})(::NeverAlias, A::Zeros)
  return copy(A)
end