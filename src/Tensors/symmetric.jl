
dims(H::Hermitian{<:Number,<:Tensor}) = dims(parent(H))

dim(H::Hermitian{<:Number,<:Tensor},
    i::Int) = dim(parent(H),i)

matrix(H::Hermitian{<:Number,<:Tensor}) = Hermitian(matrix(parent(H)))

inds(H::Hermitian{<:Number,<:Tensor}) = inds(parent(H))

ind(H::Hermitian{<:Number,<:Tensor},
    i::Int) = ind(parent(H),i)

nnzblocks(H::Hermitian{<:Number,<:Tensor}) = nnzblocks(parent(H))

nzblocks(H::Hermitian{<:Number,<:Tensor}) = nzblocks(parent(H))

nblocks(H::Hermitian{<:Number,<:Tensor}) = nblocks(parent(H))

nzblock(H::Hermitian{<:Number,<:Tensor},
        i::Int) = nzblock(parent(H), i)

blockview(H::Hermitian{<:Number,<:Tensor},
          i::Int) = Hermitian(blockview(parent(H), i))

