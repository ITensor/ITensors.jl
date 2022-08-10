"""
ITensor is a library for rapidly creating correct and efficient tensor network algorithms.

An ITensor is a tensor whose interface is independent of its memory layout. ITensor indices are 'intelligent' meaning they carry extra information and 'recognize' each other automatically when contracting or adding ITensors.

The ITensor library includes composable and extensible algorithms for optimizing and transforming tensor networks, such as matrix product state and matrix product operators.

# Example Usage

Define tensor indices i and j

    i = Index(2, "i")
    j = Index(3, "j")

Make an ITensor with these indices

    A = ITensor(i,j)

Set the i==2,j==1 element to -2.6

    A[j=>1,i=>2] = -2.6
    A[i=>2,j=>1] = -2.6 #this has the same effect

Make an ITensor with random elements

    B = randomITensor(j,i)

Add ITensors A and B together (ok that indices in different order)

    C = A + B

# Other Features of ITensor

- Tools for **tensor networks**, such as matrix product states (MPS) / tensor trains (TT)
- **Algorithms** for solving linear equations in MPS form (such as DMRG) or for integrating differential equations ("time evolving MPS")
- ITensors can have **sparse data** internally, such as block sparsity or diagonal sparsity, while having the same interface as dense ITensors
- ITensors can have **symmetry properties** (invariance or equivariance) under group transformations of the indices. In physics terminology such ITensors conserve quantum numbers.

# Documentation and Resources

ITensor website: https://itensor.org/

Documentation: https://itensor.github.io/ITensors.jl/stable/


"""
module ITensors

#####################################
# External packages
#
using Adapt
using BitIntegers
using ChainRulesCore
using Compat
using Functors
using HDF5
using IsApprox
using KrylovKit
using LinearAlgebra
using NDTensors
using PackageCompiler
using Pkg
using Printf
using Random
using SerializedElementArrays
using SnoopPrecompile
using StaticArrays
using TimerOutputs
using Zeros

#####################################
# General utility functions
#
include("utils.jl")

#####################################
# ContractionSequenceOptimization
#
include("ContractionSequenceOptimization/ContractionSequenceOptimization.jl")
using .ContractionSequenceOptimization

#####################################
# LazyApply
#
include("LazyApply/LazyApply.jl")
using .LazyApply

#####################################
# Ops
#
include("Ops/Ops.jl")
using .Ops
import .Ops: sites, name

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
# Algorithm type for selecting
# different algorithm backends
# (for internal or advanced usage)
#
include("algorithm.jl")

#####################################
# Index and IndexSet
#
# include("lastval.jl")
# include("smallstring.jl") # Not currently using in TagSet
include("readwrite.jl")
# include("not.jl")
# include("tagset.jl")
# include("arrow.jl")
# include("symmetrystyle.jl")
# include("index.jl")
# include("set_operations.jl")
# include("indexset.jl")

#####################################
# ITensor
#
include("itensor.jl")
include("broadcast.jl")
include("decomp.jl")
include("iterativesolvers.jl")
include("adapt.jl")

#####################################
# QNs
#
# include("qn/qn.jl")
# include("qn/qnindex.jl")
# include("qn/qnindexset.jl")
include("qn/qnitensor.jl")
include("nullspace.jl")

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
include("mps/adapt.jl")

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
include("physics/autompo/matelem.jl")
include("physics/autompo/qnmatelem.jl")
include("physics/autompo/opsum_to_mpo_generic.jl")
include("physics/autompo/opsum_to_mpo.jl")
include("physics/autompo/opsum_to_mpo_qn.jl")

#####################################
# Ops to MPO conversions
#
# include("Ops/ops_mpo.jl")

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

#####################################
# Deprecated
#
include("Deprecated/Deprecated.jl")

#####################################
# Precompile certain functions
#
# if Base.VERSION >= v"1.4.2"
#   include("precompile.jl")
#   _precompile_()
# end

@precompile_all_calls begin
  for eltype in (Float32, Float64, ComplexF32, ComplexF64)
    for space in (2, [QN(0) => 2, QN(1) => 2])
      i = Index(space)
      j = Index(space)
      k = Index(space)
      A = randomITensor(eltype, i, dag(j))
      B = randomITensor(eltype, j, dag(k))
      C = A * B
      C = A * dag(A)
      U, S, V = svd(A, i; maxdim=1)
      U, S, V = svd(A, i; maxdim=1, alg="recursive")
    end
    for conserve_qns in (false, true)
      n = 4
      sites = siteinds("S=1/2", n; conserve_qns)
      opsum = OpSum()
      for j in 1:(n - 1)
        opsum += "Sz", j, "Sz", j + 1
        opsum += 0.5, "S+", j, "S-", j + 1
        opsum += 0.5, "S-", j, "S+", j + 1
      end
      H = MPO(eltype, opsum, sites)
      psi0 = randomMPS(eltype, sites, j -> isodd(j) ? "↑" : "↓"; linkdims=2)
      energy, psi = dmrg(H, psi0; nsweeps=1, maxdim=10, cutoff=1e-4, outputlevel=1)
    end
  end
end

end # module ITensors
