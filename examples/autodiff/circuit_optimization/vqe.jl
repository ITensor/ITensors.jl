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

layer(N, θ⃗) = vcat(Rylayer(N, θ⃗), CXlayer(N))

# The variational circuit we want to optimize
function variational_circuit(N, nlayers, θ⃗)
  range = 1:N
  circuit = layer(N, θ⃗[range])
  for n in 1:(nlayers - 1)
    circuit = vcat(circuit, layer(N, θ⃗[range .+ n * N]))
  end
  return circuit
end

N = 4
s = siteinds("Qubit", N)

h = 1.3
ℋ = OpSum()
for j in 1:(N - 1)
  ℋ .+= -1, "Z", j, "Z", j + 1
end
for j in 1:N
  ℋ .+= h, "X", j
end
H = prod(MPO(ℋ, s))
ψ = prod(MPS(s, "0"))

N = 4
nlayers = 5

function loss(θ⃗)
  gates = variational_circuit(N, nlayers, θ⃗)
  U = buildcircuit(gates, s)
  return rayleigh_quotient(H, (U, ψ))
end

seed!(1234)
θ⃗₀ = 2π * rand(N * nlayers)
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

Uₒₚₜ = buildcircuit(variational_circuit(N, nlayers, θ⃗ₒₚₜ), s)
Uₒₚₜψ = apply(Uₒₚₜ, ψ)
@show (Uₒₚₜψ' * H * Uₒₚₜψ)[]

nothing
