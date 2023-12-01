@eval module $(gensym())
include("test_array.jl")
include("test_diagonalarray.jl")
include("test_sparsearray.jl")
end
