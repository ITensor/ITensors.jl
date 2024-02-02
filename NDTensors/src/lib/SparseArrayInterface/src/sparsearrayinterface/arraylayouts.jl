using ArrayLayouts: MemoryLayout

abstract type AbstractSparseLayout <: MemoryLayout end

struct SparseLayout <: AbstractSparseLayout end
