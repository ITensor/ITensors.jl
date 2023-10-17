using NDTensors
using LinearAlgebra
using Test

@testset "Tensor wrapping AbstractArrays" begin
  include("array.jl")
  include("blocksparsearray.jl")
end
