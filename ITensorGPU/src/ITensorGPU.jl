module ITensorGPU

using CUDA
using CUDA.Adapt
using CUDA.CUTENSOR
using CUDA.CUBLAS
using CUDA.CUSOLVER
using Functors
using LinearAlgebra
using Random, Strided
using TimerOutputs
using SimpleTraits
using StaticArrays
using ITensors
using ITensors.NDTensors
using Strided
import CUDA: CuArray, CuMatrix, CuVector, cu
import CUDA.CUTENSOR: cutensorContractionPlan_t, cutensorAlgo_t
import CUDA.Adapt: adapt_structure
import CUDA.Mem: pin
#=
const devs = Ref{Vector{CUDAdrv.CuDevice}}()
const dev_rows = Ref{Int}(0)
const dev_cols = Ref{Int}(0)
function __init__()
  voltas    = filter(dev->occursin("V100", CUDAdrv.name(dev)), collect(CUDAdrv.devices()))
  pascals    = filter(dev->occursin("P100", CUDAdrv.name(dev)), collect(CUDAdrv.devices()))
  devs[] = voltas[1:1]
  #devs[] = pascals[1:2]
  CUBLASMG.cublasMgDeviceSelect(CUBLASMG.mg_handle(), length(devs[]), devs[])
  dev_rows[] = 1
  dev_cols[] = 1
end
=#
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
import ITensors.NDTensors:
  can_contract,
  similartype,
  ContractionProperties,
  contract!!,
  _contract!!,
  _contract!,
  contract!,
  contract,
  contraction_output,
  UniformDiagTensor,
  CombinerTensor,
  contraction_output_type,
  UniformDiag,
  Diag,
  DiagTensor,
  NonuniformDiag,
  NonuniformDiagTensor,
  zero_contraction_output,
  outer!,
  outer!!,
  is_trivial_permutation,
  ind,
  permutedims!!,
  Dense,
  DenseTensor,
  Combiner,
  Tensor,
  data,
  getperm,
  compute_contraction_properties!,
  Atrans,
  Btrans,
  Ctrans,
  _contract_scalar!,
  _contract_scalar_noperm!

using ITensors.NDTensors: setdata, setstorage, cpu, IsWrappedArray, parenttype

import Base.*, Base.permutedims!
import Base: similar
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

#const ContractionPlans = Dict{String, Tuple{cutensorAlgo_t, cutensorContractionPlan_t}}()
const ContractionPlans = Dict{String,cutensorAlgo_t}()

export cu,
  cpu, cuITensor, randomCuITensor, cuMPS, randomCuMPS, productCuMPS, randomCuMPO, cuMPO

end #module
