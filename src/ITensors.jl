"""
ITensors is a library for rapidly creating correct and efficient tensor network algorithms.

An ITensor is a tensor whose interface is independent of its memory layout. ITensor indices are objects which carry extra information and which 'recognize' each other (compare equal to each other).

The ITensor library also includes composable and extensible algorithms for optimizing and transforming tensor networks, such as matrix product state and matrix product operators, such as the DMRG algorithm.
"""
module ITensors

#####################################
# External packages
#
using Compat
using HDF5
using KrylovKit
using LinearAlgebra
using NDTensors
using PackageCompiler
using Pkg
using Printf
using Random
using SerializedElementArrays
using StaticArrays
using TimerOutputs
using Zeros

#####################################
# ContractionSequenceOptimization
#
include("ContractionSequenceOptimization/ContractionSequenceOptimization.jl")
using .ContractionSequenceOptimization

#####################################
# LazyApply
#
include("LazyApply/LazyApply.jl")
using .LazyApply: Applied, Sum, ∑, Prod, ∏, Scaled, α, coefficient

#####################################
# Ops
#
include("Ops/Ops.jl")
using .Ops

#####################################
# Directory helper functions (useful for
# running examples)
#
src_dir() = dirname(pathof(@__MODULE__))
pkg_dir() = joinpath(src_dir(), "..")
examples_dir() = joinpath(pkg_dir(), "examples")

#####################################
# Determine version and uuid of the package
#
function _parse_project_toml(field::String)
  return Pkg.TOML.parsefile(joinpath(pkg_dir(), "Project.toml"))[field]
end
version() = VersionNumber(_parse_project_toml("version"))
uuid() = Base.UUID(_parse_project_toml("uuid"))

#####################################
# Exports
#
include("exports.jl")

#####################################
# Imports
#
include("imports.jl")

#####################################
# Global Variables
#
include("global_variables.jl")

#####################################
# Index and IndexSet
#
include("lastval.jl")
include("smallstring.jl") # Not currently using in TagSet
include("readwrite.jl")
include("not.jl")
include("tagset.jl")
include("arrow.jl")
include("symmetrystyle.jl")
include("index.jl")
include("set_operations.jl")
include("indexset.jl")

#####################################
# ITensor
#
include("itensor.jl")
include("broadcast.jl")
include("decomp.jl")
include("iterativesolvers.jl")

#####################################
# QNs
#
include("qn/qn.jl")
include("qn/qnindex.jl")
include("qn/qnindexset.jl")
include("qn/qnitensor.jl")

#####################################
# Ops to ITensor conversions
#
include("Ops/ops_itensor.jl")

#####################################
# MPS/MPO
#
include("mps/abstractmps.jl")
include("mps/deprecated.jl")
include("mps/mps.jl")
include("mps/mpo.jl")
include("mps/sweeps.jl")
include("mps/abstractprojmpo.jl")
include("mps/projmpo.jl")
include("mps/diskprojmpo.jl")
include("mps/projmposum.jl")
include("mps/projmps.jl")
include("mps/projmpo_mps.jl")
include("mps/observer.jl")
include("mps/dmrg.jl")

#####################################
# Physics
#
include("physics/sitetype.jl")
include("physics/lattices.jl")
include("physics/site_types/generic_sites.jl")
include("physics/site_types/qubit.jl")
include("physics/site_types/spinhalf.jl")
include("physics/site_types/spinone.jl")
include("physics/site_types/fermion.jl")
include("physics/site_types/electron.jl")
include("physics/site_types/tj.jl")
include("physics/site_types/qudit.jl") # EXPERIMENTAL
include("physics/site_types/boson.jl") # EXPERIMENTAL
include("physics/fermions.jl")
include("physics/autompo.jl")

#####################################
# Ops to MPO conversions
#
include("Ops/ops_mpo.jl")

#####################################
# Trotter-Suzuki decomposition
#
include("Ops/trotter.jl")

#####################################
# ITensorChainRules
#
include("ITensorChainRules/ITensorChainRules.jl")

#####################################
# ITensorNetworkMaps
#
include("ITensorNetworkMaps/ITensorNetworkMaps.jl")

#####################################
# ITensorVisualizationCore
#
include("ITensorVisualizationCore/ITensorVisualizationCore.jl")
using .ITensorVisualizationCore

#####################################
# Deprecations
#
include("deprecated.jl")

#####################################
# Argument parsing
#
include("argsdict/argsdict.jl")

#####################################
# Package compilation
#
include("packagecompile/compile.jl")

#####################################
# Developer tools, for internal
# use only
#
include("developer_tools.jl")

function __init__()
  return resize!(empty!(INDEX_ID_RNGs), Threads.nthreads()) # ensures that we didn't save a bad object
end

#####################################
# Precompile certain functions
#
#if Base.VERSION >= v"1.4.2"
#  include("precompile.jl")
#  _precompile_()
#end

end # module ITensors
