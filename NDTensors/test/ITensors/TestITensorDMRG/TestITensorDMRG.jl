## This is getting closer still working on it. No need to review
## Failing for CUDA mostly with eigen (I believe there is some noise in
## eigen decomp with CUBLAS to give slightly different answer than BLAS)
module TestITensorDMRG
using ITensors
using NDTensors
using NDTensors.AMDGPUExtensions: roc
using NDTensors.CUDAExtensions: cu
using NDTensors.MetalExtensions: mtl
using Random

reference_energies = Dict([
  (4, -1.6160254037844384), (8, -3.374932598687889), (10, -4.258035207282885)
])

is_broken(dev, elt::Type, conserve_qns::Val) = false
## Currently there is an issue in blocksparse cutensor, seems to be related to using @view, I am still working to fix this issue.
## For some reason ComplexF64 works and throws an error in `test_broken`
is_broken(dev::typeof(cu), elt::Type, conserve_qns::Val{true}) = ("cutensor" ∈ ARGS && elt != ComplexF64)

include("dmrg.jl")

end
