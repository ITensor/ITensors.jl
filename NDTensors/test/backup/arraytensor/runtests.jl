@eval module $(gensym())
using Test: @testset
@testset "Tensor wrapping AbstractArrays $(f)" for f in [
  "array.jl", "blocksparsearray.jl", "diagonalarray.jl"
]
  include(f)
end
end
