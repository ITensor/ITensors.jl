struct Zeros{ElT,N,DataT} <: AbstractArray{ElT,N}
  z::FillArrays.Zeros
  function NDTensors.Zeros{ElT,N,DataT}(inds::Tuple) where {ElT,N,DataT}
    @assert eltype(DataT) == ElT
    @assert ndims(DataT) == N
    z = FillArrays.Zeros(ElT, dim(inds))
    return new{ElT,N,DataT}(z)
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

Base.convert(x::Type{T}, z::NDTensors.Zeros) where {T<:Array} = Base.convert(x, z.z)
Base.getindex(a::Zeros, i) = Base.getindex(a.z, i)
Base.sum(z::Zeros) = sum(z.z)
LinearAlgebra.norm(z::Zeros) = norm(z.z)
setindex!(A::NDTensors.Zeros, v, I) = setindex!(A.z, v, I)

NDTensors.set_parameter(::Type{<:Zeros}, ::Position{1}, P1) = Zeros{P1}
function NDTensors.set_parameter(::Type{<:Zeros{<:Any,P2}}, ::Position{1}, P1) where {P2}
  return Zeros{P1,P2}
end
function NDTensors.set_parameter(
  ::Type{<:Zeros{<:Any,P2,P3}}, ::Position{1}, P1
) where {P2,P3<:AbstractArray}
  P = NDTensors.set_parameter(P3, Position(1), P1)
  return Zeros{P1,P2,P}
end
NDTensors.set_parameter(::Type{<:Zeros}, ::Position{2}, P2) = Zeros{<:Any,P2}
NDTensors.set_parameter(::Type{<:Zeros{P1}}, ::Position{2}, P2) where {P1} = Zeros{P1,P2}
function NDTensors.set_parameter(
  ::Type{<:Zeros{P1,<:Any,P3}}, ::Position{2}, P2
) where {P1,P3<:AbstractArray}
  return Zeros{P1,P2,NDTensors.set_parameter(P3, Position(1), P1)}
end
function NDTensors.set_parameter(::Type{<:Zeros{P1,P2}}, ::Position{3}, P3) where {P1,P2}
  return Zeros{P1,P2,NDTensors.set_parameter(P3, Position(1), P1)}
end
