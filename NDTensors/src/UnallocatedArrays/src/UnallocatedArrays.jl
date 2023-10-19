module UnallocatedArrays

using FillArrays
using LinearAlgebra

include("unallocatedfill.jl")
include("unallocatedzeros.jl")

export UnallocatedFill, UnallocatedZeros
end
