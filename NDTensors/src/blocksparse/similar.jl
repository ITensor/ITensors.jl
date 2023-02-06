# NDTensors.similar
function similar(storagetype::Type{<:BlockSparse}, blockoffsets::BlockOffsets, dims::Tuple)
  # TODO: Don't convert to an `AbstractVector` with `vec`, once we support
  # more general data types.
  # data = similar(datatype(storagetype), dims)
  data = vec(similar(datatype(storagetype), dims))
  return BlockSparse(data, blockoffsets)
end

# NDTensors.similar
function similar(storagetype::Type{<:BlockSparse}, dims::Tuple)
  return error("Not implemented, must specify block offsets as well")
end

# NDTensors.similar
function similar(storagetype::Type{<:BlockSparse}, dims::Dims)
  return error("Not implemented, must specify block offsets as well")
end

## TODO: Is there a way to make this generic?
# NDTensors.similar
function similar(tensortype::Type{<:BlockSparseTensor}, blockoffsets::BlockOffsets, dims::Tuple)
  return Tensor(similar(storagetype(tensortype), blockoffsets, dims), dims)
end

# NDTensors.similar
function similar(tensor::BlockSparseTensor, blockoffsets::BlockOffsets, dims::Tuple)
  return similar(typeof(tensor), blockoffsets, dims)
end

## similar(D::BlockSparse) = setdata(D, similar(data(D)))
## 
## # TODO: test this function
## similar(D::BlockSparse, ::Type{ElT}) where {ElT} = setdata(D, similar(data(D), ElT))
## 
## function similartype(::Type{StoreT}, ::Type{ElT}) where {StoreT<:BlockSparse,ElT}
##   return BlockSparse{ElT,similartype(datatype(StoreT), ElT),ndims(StoreT)}
## end
## 
## function similar(
##   ::BlockSparseTensor{ElT,N}, blockoffsets::BlockOffsets{N}, inds
## ) where {ElT,N}
##   return BlockSparseTensor(ElT, undef, blockoffsets, inds)
## end
## 
## function similar(
##   ::Type{<:BlockSparseTensor{ElT,N}}, blockoffsets::BlockOffsets{N}, inds
## ) where {ElT,N}
##   return BlockSparseTensor(ElT, undef, blockoffsets, inds)
## end
## 
## # This version of similar creates a tensor with no blocks
## function similar(::Type{TensorT}, inds::Tuple) where {TensorT<:BlockSparseTensor}
##   return similar(TensorT, BlockOffsets{ndims(TensorT)}(), inds)
## end
## 
## # Special version for BlockSparseTensor
## # Generic version doesn't work since BlockSparse us parametrized by
## # the Tensor order
## function similartype( 
##   ::Type{<:Tensor{ElT,NT,<:BlockSparse{ElT,VecT},<:Any}}, indsR
## ) where {NT,ElT,VecT}
##   NR = length(indsR)
##   return Tensor{ElT,NR,BlockSparse{ElT,VecT,NR},typeof(indsR)}
## end
## 
## function similartype(
##   ::Type{<:Tensor{ElT,NT,<:BlockSparse{ElT,VecT},<:Any}}, indsR
## ) where {NT,ElT,VecT,IndsR<:NTuple{NR}} where {NR}
##   return Tensor{ElT,NR,BlockSparse{ElT,VecT,NR},typeof(indsR)}
## end
## 
## similar(D::DiagBlockSparse, n::Int) = setdata(D, similar(data(D), n))
## 
## function similar(D::DiagBlockSparse, ::Type{ElR}, n::Int) where {ElR}
##   return setdata(D, similar(data(D), ElR, n))
## end
## 
## # TODO: write in terms of ::Int, not inds
## similar(D::NonuniformDiagBlockSparse) = setdata(D, similar(data(D)))
## 
## similar(D::NonuniformDiagBlockSparse, ::Type{S}) where {S} = setdata(D, similar(data(D), S))
## #similar(D::NonuniformDiagBlockSparse,inds) = DiagBlockSparse(similar(data(D),minimum(dims(inds))), diagblockoffsets(D))
## #function similar(D::Type{<:NonuniformDiagBlockSparse{ElT,VecT}},inds) where {ElT,VecT}
## #  return DiagBlockSparse(similar(VecT,diaglength(inds)), diagblockoffsets(D))
## #end
## 
## similar(D::UniformDiagBlockSparse) = setdata(D, zero(eltype(D)))
## similar(D::UniformDiagBlockSparse, inds) = similar(D)
## function similar(::Type{<:UniformDiagBlockSparse{ElT}}, inds) where {ElT}
##   return DiagBlockSparse(zero(ElT), diagblockoffsets(D))
## end
## 
## # Needed to get slice of DiagBlockSparseTensor like T[1:3,1:3]
## function similar(
##   T::DiagBlockSparseTensor{<:Number,N}, ::Type{ElR}, inds::Dims{N}
## ) where {ElR<:Number,N}
##   return tensor(similar(storage(T), ElR, minimum(inds)), inds)
## end
