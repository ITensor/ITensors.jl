module UnspecifiedTypes

using LinearAlgebra

include("unspecifiednumber.jl")
include("unspecifiedzero.jl")

include("unspecifiedarray.jl")

export UnspecifiedArray, UnspecifiedNumber, UnspecifiedZero
end
