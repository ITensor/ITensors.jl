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

default_rtol(elt::Type) = 10^(0.75 * log10(eps(real(elt))))

is_supported_eltype(dev, elt::Type) = true
is_supported_eltype(dev::typeof(NDTensors.mtl), elt::Type{Float64}) = false
is_supported_eltype(dev::typeof(NDTensors.mtl), elt::Type{<:Complex}) = is_supported_eltype(dev, real(elt))

include("dmrg.jl")

end
