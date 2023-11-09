module UnallocatedArrays
using FillArrays
using LinearAlgebra
using NDTensors.SetParameters

include("import.jl")

include("unallocatedfill.jl")
include("unallocatedzeros.jl")
include("set_types.jl")
include("unallocated_impl.jl")

export UnallocatedFill, UnallocatedZeros, alloctype, set_alloctype, allocate
end
