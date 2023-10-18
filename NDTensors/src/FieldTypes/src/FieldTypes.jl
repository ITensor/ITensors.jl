module FieldTypes

using FillArrays
using LinearAlgebra

#include("SetParameters/src/SetParameters.jl")
#using ..SetParameters

include("unspecifiednumber/unspecifiednumber.jl")
include("unspecifiednumber/unspecifiedzero.jl")

include("unallocatedarray/unallocatedfill.jl")
include("unallocatedarray/unallocatedzeros.jl")

include("unspecifiedarray/unspecifiedarray.jl")


export UnallocatedFill,
 UnallocatedZeros,
 UnspecifiedArray,
 UnspecifiedNumber,
 UnspecifiedZero
end