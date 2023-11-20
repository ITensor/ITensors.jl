module UnallocatedArrays
using FillArrays
using LinearAlgebra
using NDTensors.SetParameters

include("import.jl")

include("abstractunallocatedarray.jl")
include("unallocatedfill.jl")
include("unallocatedzeros.jl")
include("set_types.jl")

export UnallocatedFill, UnallocatedZeros, alloctype, set_alloctype, allocate
end
