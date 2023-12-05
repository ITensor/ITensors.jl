using ArrayLayouts: ArrayLayouts
using ..SparseArrayInterface: AbstractSparseLayout

abstract type AbstractDiagonalLayout <: AbstractSparseLayout end
struct DiagonalLayout <: AbstractDiagonalLayout end

ArrayLayouts.MemoryLayout(::Type{<:AbstractDiagonalArray}) = DiagonalLayout()
