## This is getting closer still working on it. No need to review
## Failing for CUDA mostly with eigen (I believe there is some noise in
## eigen decomp with CUBLAS to give slightly different answer than BLAS)
module TestITensorDMRG
using ITensors
using NDTensors

reference_energies = Dict([(4, -1.6160254037844384), (8, -3.374932598687889)])

function get_ref_value(device, sites, elt)
  return reference_energies[sites]
end

include("dmrg.jl")
end
