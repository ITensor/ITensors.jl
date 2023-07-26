struct Zeros{ElT, N, DataT} <:AbstractArray{ElT,N}
  z::FillArrays.Zeros
  function NDTensors.Zeros{ElT, N, DataT}(inds::Tuple) where {ElT, N, DataT}
    @assert eltype(DataT) == ElT
    @assert ndims(DataT) == N
    z = FillArrays.Zeros(ElT, dim(inds))
    new{ElT, N, DataT}(z)
  end
end

Base.ndims(::NDTensors.Zeros{ElT, N}) where {ElT, N} = N
ndims(::NDTensors.Zeros{ElT, N}) where {ElT, N} = N
Base.eltype(::Zeros{ElT}) where {ElT} = ElT
datatype(::NDTensors.Zeros{ElT, N, DataT}) where{ElT, N, DataT} = DataT
datatype(::Type{<:NDTensors.Zeros{ElT, N, DataT}}) where{ElT, N, DataT} = DataT

Base.size(zero::Zeros) = Base.size(zero.z)

Base.print_array(io::IO, X::Zeros) = Base.print_array(io, X.z)

data(zero::Zeros) = zero.z
getindex(zero::Zeros) = getindex(zero.z)

array(zero::Zeros) = datatype(zero)(zero.z)

Base.convert(x::Type{T}, z::NDTensors.Zeros) where {T<:Array}= Base.convert(x, z.z)