using ITensors
using OptimKit
using Zygote

function ising(n; J, h)
  os = OpSum()
  for j in 1:(n - 1)
    os += -J, "Z", j, "Z", j + 1
  end
  for j in 1:n
    os += -h, "X", j
  end
  return os
end

function loss(H, ψ)
  n = length(ψ)
  ψHψ = ITensor(1.0)
  ψψ = ITensor(1.0)
  for j in 1:n
    ψHψ = ψHψ * dag(ψ[j]') * H[j] * ψ[j]
    ψψ = ψψ * replaceinds(dag(ψ[j]'), s[j]' => s[j]) * ψ[j]
  end
  return ψHψ[] / ψψ[]
end

n = 10
s = siteinds("S=1/2", n)
J = 1.0
h = 0.5

# Loss function only works with `Vector{ITensor}`,
# extract with `ITensors.data`.
ψ0 = ITensors.data(randomMPS(s; linkdims=10))
H = ITensors.data(MPO(ising(n; J, h), s))

loss(ψ) = loss(H, ψ)

optimizer = LBFGS(; maxiter=25, verbosity=2)
function loss_and_grad(x)
  y, (∇,) = withgradient(loss, x)
  return y, ∇
end
ψ, fs, gs, niter, normgradhistory = optimize(loss_and_grad, ψ0, optimizer)
Edmrg, ψdmrg = dmrg(MPO(H), MPS(ψ0); nsweeps=10, cutoff=1e-8)

@show loss(ψ0), norm(loss'(ψ0))
@show loss(ψ), norm(loss'(ψ))
@show loss(ITensors.data(ψdmrg)), norm(loss'(ITensors.data(ψdmrg)))
