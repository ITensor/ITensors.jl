using ITensors
using LinearAlgebra
using Zygote
using Random

Random.seed!(1234)

function heisenberg_hamiltonian(N)
  os = OpSum()
  for j in 1:(N - 1)
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
    os += "Sz", j, "Sz", j + 1
  end
  return os
end

# Number of sites in the system
N = 6
s = siteinds("S=1/2", N)
os = heisenberg_hamiltonian(N)
Hmpo = MPO(os, s)

H = prod(Hmpo)

# Number of eigenstates to target
p = 3

D, V = eigen(H; ishermitian=true)
exact_energies = sort(ITensors.data(D))[1:p]

n = isqrt(dim(H))
sr = Index(p, "renorm")
X₀ = randomITensor(s..., sr)
U₀, _ = polar(X₀, s)
U₀ = noprime(U₀)

# Loss function for low energy eigenspace.
# Reference: https://arxiv.org/abs/2007.01287, page 9, example A
function E(H, U)
  UHU = dag(U) * H * U'
  trUHU = (UHU * dag(δ(inds(UHU))))[]
  return trUHU
end

loss(U) = E(H, U)

@show sum(exact_energies)
@show loss(U₀)

# Step size
η = 0.1

# Move to another point of the Stiefel manifold using the
# SVD retraction formula.
# Reference: https://arxiv.org/abs/2007.01287, page 8, equation (17)
RSVD(X, W, is) = noprime(polar(X + W, is)[1])

# Riemannian optimization of Stiefel Manifold (manifold of isometric matrices)
# Reference: https://arxiv.org/abs/2007.01287, page 8, algorithm steps 3-7
function update(U₀)
  ∇ = loss'(U₀)
  U₀∇ = dag(U₀) * prime(∇; inds=sr)
  ∇U₀ = swapprime(dag(U₀∇), 0 => 1)
  ∇ᴿ = ∇ - 1 / 2 * noprime(U₀ * (U₀∇ + ∇U₀))
  U = RSVD(U₀, -η * ∇ᴿ, s)
  return U
end

function update(U, N)
  for n in 1:N
    U = update(U)
    @show n, loss(U)
  end
  return U
end

U = U₀
U = update(U, 50)

@show loss(U)

# Compare against ED
D, _ = eigen(H; ishermitian=true)
energy_exact = sum(n -> D[n, n], 1:p)
@show loss(U), energy_exact
