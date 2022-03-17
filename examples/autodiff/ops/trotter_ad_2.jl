using ITensors
using Zygote
using OptimKit

function ising(n; h)
  ℋ = Sum{Op}()
  for j in 1:(n - 1)
    ℋ -= "Z", j, "Z", j + 1
  end
  for j in 1:n
    ℋ += h, "X", j
  end
  return ℋ
end

n = 4
s = siteinds("S=1/2", n)

β = 1.0
hᶠ = 1.1

ℋᶠ = ising(n; h=hᶠ)
𝒰ᶠ = exp(-β * ℋᶠ; alg=Trotter{1}(5))

Uᶠ = Prod{ITensor}(𝒰ᶠ, s)

ψ = prod(MPS(s, "0"))

# Target state
Uᶠψ = Uᶠ(ψ)

function loss(h)
  ℋ = ising(n; h=h[1])
  𝒰ʰ = exp(-β * ℋ; alg=Trotter{1}(5))
  Uʰ = Prod{ITensor}(𝒰ʰ, s)
  Uʰψ = Uʰ(ψ)
  return -abs(inner(Uᶠψ, Uʰψ))^2 / (norm(Uᶠψ) * norm(Uʰψ))^2
end

h⁰ = [0.0]
@show loss(h⁰)
@show loss'(h⁰)

@show loss(hᶠ)
@show loss'(hᶠ)

loss_∇loss(h) = (loss(h), convert(Vector, loss'(h)))
algorithm = LBFGS(; gradtol=1e-3, verbosity=2)
hᵒᵖᵗ, _ = optimize(loss_∇loss, h⁰, algorithm)

@show loss(hᵒᵖᵗ)
@show loss'(hᵒᵖᵗ)
