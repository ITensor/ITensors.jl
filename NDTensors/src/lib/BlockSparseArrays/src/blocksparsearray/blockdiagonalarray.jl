using LinearAlgebra: Diagonal

const BlockDiagonal{T,A,Axes,V<:AbstractVector{A}} = BlockSparseMatrix{T,A,Diagonal{A,V},Axes}

function BlockDiagonal(blocks::AbstractVector{<:AbstractMatrix})
    return BlockSparseArray(Diagonal(blocks), (blockedrange(size.(blocks,1)), blockedrange(size.(blocks,2))))
end