module UnallocatedArrays
include("abstractfill/abstractfill.jl")

include("unallocatedfill.jl")
include("unallocatedzeros.jl")
include("broadcast.jl")
include("abstractunallocatedarray.jl")
include("set_types.jl")

export UnallocatedFill, UnallocatedZeros, alloctype, set_alloctype, allocate
end
