module ITensorGaussianMPS

using Compat
using ITensors
using ITensors.NDTensors
using LinearAlgebra

import LinearAlgebra: Givens

export slater_determinant_to_mps,
  slater_determinant_to_gmps,
  slater_determinant_to_gmera,
  hopping_hamiltonian,
  slater_determinant_matrix,
  retarded_green_function,
  lesser_green_function,
  greater_green_function

include("gmps.jl")
include("gmera.jl")
include("dynamics.jl")

end
