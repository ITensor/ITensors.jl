# NDTensors.similar
function similar(storagetype::Type{<:BlockSparse}, blockoffsets::BlockOffsets, dims::Tuple)
  data = similar(datatype(storagetype), nnz(blockoffsets, dims))
  return BlockSparse(data, blockoffsets)
end

# NDTensors.similar
function similar(storagetype::Type{<:BlockSparse}, dims::Tuple)
  # Create an empty BlockSparse storage
  return similartype(storagetype, dims)()
end

# NDTensors.similar
function similar(storagetype::Type{<:BlockSparse}, dims::Dims)
  # Create an empty BlockSparse storage
  return similartype(storagetype, dims)()
end

## TODO: Is there a way to make this generic?
# NDTensors.similar
function similar(
  tensortype::Type{<:BlockSparseTensor}, blockoffsets::BlockOffsets, dims::Tuple
)
  return Tensor(similar(storagetype(tensortype), blockoffsets, dims), dims)
end

# NDTensors.similar
function similar(tensor::BlockSparseTensor, blockoffsets::BlockOffsets, dims::Tuple)
  return similar(typeof(tensor), blockoffsets, dims)
end

## ## TODO: Determine if the methods below are needed.
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
