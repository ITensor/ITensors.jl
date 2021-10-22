using ITensors
using OptimKit
using Zygote
using Random: seed!

include("circuit.jl")

function gate(::Val{:Ry}; θ)
  return [
    cos(θ / 2) -sin(θ / 2)
    sin(θ / 2) cos(θ / 2)
  ]
end

function gate(::Val{:CX})
  return [
    1 0 0 0
    0 1 0 0
    0 0 0 1
    0 0 1 0
  ]
end

function Rylayer(N, θ⃗)
  return [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
end

function CXlayer(N)
  return [("CX", (n, n + 1)) for n in 1:2:(N - 1)]
end

# The variational circuit we want to optimize
function variational_circuit(θ⃗)
  N = length(θ⃗)
  return vcat(Rylayer(N, θ⃗), CXlayer(N), Rylayer(N, θ⃗), CXlayer(N))
end

N = 4
θ⃗ = 2π .* rand(N)
gates = variational_circuit(θ⃗)

s = siteinds("Qubit", N)
ψₘₚₛ = MPS(s, "0")
ψ = prod(ψₘₚₛ)
U = buildcircuit(gates, s)
# Create the target state
Uψ = apply(U, ψ)

@show inner_circuit(Uψ, U, ψ);

function loss(θ⃗)
  gates = variational_circuit(θ⃗)
  U = buildcircuit(gates, s)
  return -abs(inner_circuit(Uψ, U, ψ))^2
end

seed!(1234)
θ⃗₀ = randn!(copy(θ⃗))
@show θ⃗
@show loss(θ⃗)
@show loss'(θ⃗)
@show θ⃗₀
@show loss(θ⃗₀)
@show loss'(θ⃗₀)

fg(x) = (loss(x), convert(Vector, loss'(x)))
θ⃗ₒₚₜ, fₒₚₜ, gₒₚₜ, numfg, normgradhistory = optimize(fg, θ⃗₀, GradientDescent())
@show θ⃗ₒₚₜ
@show fₒₚₜ
@show gₒₚₜ
@show numfg
println("normgradhistory = ")
display(normgradhistory)

nothing
