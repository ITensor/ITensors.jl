module ITensorGaussianMPS

using Compat
using ITensors
using ITensors.NDTensors
using LinearAlgebra

import LinearAlgebra: Givens

export slater_determinant_to_mps,
  slater_determinant_to_gmps, hopping_hamiltonian, slater_determinant_matrix

export G_R, G_G, G_L

include("gmps.jl")
include("dynamics.jl")

end
