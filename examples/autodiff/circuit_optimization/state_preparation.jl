using ITensors
using Random
using Zygote

# The variational circuit we want to optimize
variational_circuit(n, θ⃗) = 
  [[("Ry", (j,), (θ = θ⃗[j],)) for j in 1:n];
   [("CX", (j,j+1)) for j in 1:2:n-1]]

Random.seed!(1234)
nsites = 4
θ⃗ = 2π .* rand(nsites)

θ⃗ᵗᵃʳᵍᵉᵗ = 2π * rand(nsites)
𝒰ᵗᵃʳᵍᵉᵗ = variational_circuit(nsites, θ⃗ᵗᵃʳᵍᵉᵗ)

s = siteinds("Qubit", nsites)
Uᵗᵃʳᵍᵉᵗ = ops(𝒰ᵗᵃʳᵍᵉᵗ, s)

ψ0 = MPS(s, "0")

ψᵗᵃʳᵍᵉᵗ = apply(Uᵗᵃʳᵍᵉᵗ, ψ0; cutoff=1e-8)

ψ0 = prod(ψ0)
ψᵗᵃʳᵍᵉᵗ = prod(ψᵗᵃʳᵍᵉᵗ)

function loss(θ⃗)
  𝒰θ⃗ = variational_circuit(nsites, θ⃗)
  Uθ⃗ = ops(𝒰θ⃗, s)
  ψθ⃗ = apply(Uθ⃗, ψ0)
  ip = (ψᵗᵃʳᵍᵉᵗ * ψθ⃗)[]
  return -abs2(ip)
end

@show loss(θ⃗), loss(θ⃗ᵗᵃʳᵍᵉᵗ)
@show loss'(θ⃗)

