using ArrayLayouts: ArrayLayouts
using ..SparseArraysBase: AbstractSparseLayout

abstract type AbstractDiagonalLayout <: AbstractSparseLayout end
struct DiagonalLayout <: AbstractDiagonalLayout end

ArrayLayouts.MemoryLayout(::Type{<:AbstractDiagonalArray}) = DiagonalLayout()
