
# XXX Should this permute the dimensions?
dims(H::Hermitian{<:Number,<:Tensor}) = dims(parent(H))

# XXX Should this permute the dimensions?
dim(H::Hermitian{<:Number,<:Tensor}, i::Int) = dim(parent(H), i)

matrix(H::Hermitian{<:Number,<:Tensor}) = Hermitian(matrix(parent(H)))

# XXX Should this permute the indices?
inds(H::Hermitian{<:Number,<:Tensor}) = inds(parent(H))

# XXX Should this permute the indices?
ind(H::Hermitian{<:Number,<:Tensor}, i::Int) = ind(parent(H), i)

# XXX Should this tranpose the block locations?
nnzblocks(H::Hermitian{<:Number,<:Tensor}) = nnzblocks(parent(H))

# XXX Should this tranpose the block locations?
nzblocks(H::Hermitian{<:Number,<:Tensor}) = nzblocks(parent(H))

# XXX Should this tranpose the block locations?
eachnzblock(H::Hermitian{<:Number,<:Tensor}) = eachnzblock(parent(H))

nblocks(H::Hermitian{<:Number,<:Tensor}) = nblocks(parent(H))

blockview(H::Hermitian{<:Number,<:Tensor}, block) = Hermitian(blockview(parent(H), block))
