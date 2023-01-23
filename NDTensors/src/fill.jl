# This is a file to write generic fills for NDTensors.
#  This includes random fills, zeros, ...

function randn(StoreT::Type{<:Dense{ElT, VecT}}, dim::Integer) where {VecT<:AbstractArray{ElT}} where {ElT<:Number}
    return StoreT(randn(ElT, dim))
end



function zeros(datatype::Type{<:AbstractArray{ElT}}, dim::Integer = 0) where {ElT<:Number}
  fill!(datatype(undef, dim), zero(ElT))
end

function zeros(datatype::Type{<:AbstractArray}, dim::Integer = 0)
    zeros(datatype{default_eltype()}, dim)
end

zeros(DenseT::Type{<:Dense}, inds) = zeros(DenseT, dim(inds))

# Generic for handling `Vector` and `CuVector`
function zeros(storagetype::Type{<:Dense}, dim::Int)
  vector = zeros(datatype(storagetype), dim)
  return storagetype(vector)
end