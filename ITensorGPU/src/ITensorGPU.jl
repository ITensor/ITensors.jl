module ITensorGPU
using NDTensors

using Adapt
using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using Functors
using ITensors
using LinearAlgebra
using Random
using SimpleTraits
using StaticArrays
using Strided
using TimerOutputs
using cuTENSOR

using NDTensors: setdata, setstorage, cpu, IsWrappedArray, parenttype

import Adapt: adapt_structure
import Base: *, permutedims!
import CUDA: CuArray, CuMatrix, CuVector, cu
import CUDA.Mem: pin
import ITensors:
  randn!,
  compute_contraction_labels,
  eigen,
  tensor,
  scale!,
  unioninds,
  array,
  matrix,
  vector,
  polar,
  tensors,
  truncate!,
  leftlim,
  rightlim,
  permute,
  BroadcastStyle,
  Indices
import NDTensors:
  Atrans,
  Btrans,
  CombinerTensor,
  ContractionProperties,
  Combiner,
  Ctrans,
  Diag,
  DiagTensor,
  Dense,
  DenseTensor,
  NonuniformDiag,
  NonuniformDiagTensor,
  Tensor,
  UniformDiag,
  UniformDiagTensor,
  _contract!!,
  _contract!,
  _contract_scalar!,
  _contract_scalar_noperm!,
  can_contract,
  compute_contraction_properties!,
  contract!!,
  contract!,
  contract,
  contraction_output,
  contraction_output_type,
  data,
  getperm,
  ind,
  is_trivial_permutation,
  outer!,
  outer!!,
  permutedims!!,
  set_eltype,
  set_ndims,
  similartype,
  zero_contraction_output
import cuTENSOR: cutensorContractionPlan_t, cutensorAlgo_t

#const ContractionPlans = Dict{String, Tuple{cutensorAlgo_t, cutensorContractionPlan_t}}()
const ContractionPlans = Dict{String,cutensorAlgo_t}()

include("cuarray/set_types.jl")
include("traits.jl")
include("adapt.jl")
include("tensor/cudense.jl")
include("tensor/dense.jl")
include("tensor/culinearalgebra.jl")
include("tensor/cutruncate.jl")
include("tensor/cucombiner.jl")
include("tensor/cudiag.jl")
include("cuitensor.jl")
include("mps/cumps.jl")

export cu,
  cpu, cuITensor, randomCuITensor, cuMPS, randomCuMPS, productCuMPS, randomCuMPO, cuMPO

## TODO: Is this needed?
## const devs = Ref{Vector{CUDAdrv.CuDevice}}()
## const dev_rows = Ref{Int}(0)
## const dev_cols = Ref{Int}(0)
## function __init__()
##   voltas    = filter(dev->occursin("V100", CUDAdrv.name(dev)), collect(CUDAdrv.devices()))
##   pascals    = filter(dev->occursin("P100", CUDAdrv.name(dev)), collect(CUDAdrv.devices()))
##   devs[] = voltas[1:1]
##   #devs[] = pascals[1:2]
##   CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), length(devs[]), devs[])
##   dev_rows[] = 1
##   dev_cols[] = 1
## end

end #module
