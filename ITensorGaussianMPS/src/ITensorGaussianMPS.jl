module ITensorGaussianMPS

using Compat
using ITensors
using ITensors.NDTensors
using LinearAlgebra
import LinearAlgebra: Givens

export slater_determinant_to_mps,
  correlation_matrix_to_mps,
  slater_determinant_to_gmps,
  correlation_matrix_to_gmps,
  hopping_hamiltonian,
  hopping_operator,
  quadratic_hamiltonian,
  quadratic_operator,
  slater_determinant_matrix,
  slater_determinant_to_gmera,
  eigen_gaussian

include("gmps.jl")
include("gmera.jl")
include("linalg.jl")

end
