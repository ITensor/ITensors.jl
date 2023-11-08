## This is getting closer still working on it. No need to review
## Failing for CUDA mostly with eigen (I believe there is some noise in
## eigen decomp with CUBLAS to give slightly different answer than BLAS)
module TestITensorDMRG
using Test
using ITensors
using NDTensors
using Random

reference_energies = Dict([
  (4, -1.6160254037844384), (8, -3.374932598687889), (10, -4.258035207282885)
])
test_tolerance = Dict(([Float32, 1e-5], [Float64, 1e-12]))

include("dmrg.jl")
end
