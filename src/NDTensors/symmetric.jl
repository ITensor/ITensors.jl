
dims(H::Hermitian{<:Number,<:Tensor}) = dims(parent(H))

blockdims(H::Hermitian{<:Number,<:Tensor}, b) = blockdims(parent(H), b)

dim(H::Hermitian{<:Number,<:Tensor}, i::Int) = dim(parent(H), i)

matrix(H::Hermitian{<:Number,<:Tensor}) = Hermitian(matrix(parent(H)))

inds(H::Hermitian{<:Number,<:Tensor}) = inds(parent(H))

ind(H::Hermitian{<:Number,<:Tensor}, i::Int) = ind(parent(H), i)

nnzblocks(H::Hermitian{<:Number,<:Tensor}) = nnzblocks(parent(H))

nzblocks(H::Hermitian{<:Number,<:Tensor}) = nzblocks(parent(H))

eachnzblock(H::Hermitian{<:Number,<:Tensor}) = eachnzblock(parent(H))

eachblock(H::Hermitian{<:Number,<:Tensor}) = eachblock(parent(H))

eachdiagblock(H::Hermitian{<:Number,<:Tensor}) = eachdiagblock(parent(H))

nblocks(H::Hermitian{<:Number,<:Tensor}) = nblocks(parent(H))

function blockview(H::Hermitian{<:Number,<:Tensor}, block)
  return _blockview(H, blockview(parent(H), block))
end
_blockview(::Hermitian{<:Number,<:Tensor}, blockviewH) = Hermitian(blockviewH)
_blockview(::Hermitian{<:Number,<:Tensor}, ::Nothing) = nothing
