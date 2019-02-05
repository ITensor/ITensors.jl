
module CuITensors

using CuArrays
using CuArrays.CUBLAS
using CuArrays.CUSOLVER
using LinearAlgebra
using CuTensor
using ..ITensors
using ..ITensors: CProps, Atrans, Btrans, Ctrans, truncate!
import ITensors.randn!, ITensors.storage_add!, ITensors.storage_permute!, ITensors.storage_polar, ITensors.storage_qr, ITensors.storage_eigen, ITensors.storage_svd, ITensors.qr!, ITensors.contract!, ITensors.contract_scalar!, ITensors.contract, ITensors.storage_contract, ITensors.compute_contraction_labels, ITensors.is_outer, ITensors.contract_inds, ITensors.arrtype
import Base.*
include("storage/cudense.jl")
include("storage/cucontract.jl")
include("cuitensor.jl")
include("mps/cumps.jl")
include("mps/cumpo.jl")
export cuITensor,
       randomCuITensor,
       cuMPS,
       randomCuMPS,
       cuMPO

end #module
