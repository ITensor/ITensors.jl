module UnallocatedArrays
using FillArrays: FillArrays, AbstractFill, Fill, Zeros, getindex_value
using LinearAlgebra
using NDTensors.SetParameters

include("abstractfill/abstractfill.jl")
include("abstractfill/set_types.jl")

include("unallocatedfill.jl")
include("unallocatedzeros.jl")
include("set_types.jl")

export UnallocatedFill, UnallocatedZeros, alloctype, set_alloctype, allocate
end
