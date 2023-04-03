using ITensorGaussianMPS
using LinearAlgebra
using Test

@testset "ITensorGaussianMPS.jl" begin
  include("gmps.jl")
  include("electron.jl")
  include("linalg.jl")
end
