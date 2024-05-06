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

h = 1.1

ℋ = ising(n; h)
βᶠ = 1.0
𝒰 = exp(-βᶠ * ℋ; alg=Trotter{1}(5))

U = Prod{ITensor}(𝒰, s)

ψ = prod(MPS(s, "0"))

# Target state
Uψ = U(ψ)

function loss(β)
  𝒰ᵝ = exp(-β[1] * ℋ; alg=Trotter{1}(5))
  Uᵝ = Prod{ITensor}(𝒰ᵝ, s)
  Uᵝψ = Uᵝ(ψ)
  return -abs(inner(Uψ, Uᵝψ))^2 / (norm(Uψ) * norm(Uᵝψ))^2
end

β⁰ = [0.0]
@show loss(β⁰)
@show loss'(β⁰)

@show loss(βᶠ)
@show loss'(βᶠ)

loss_∇loss(β) = (loss(β), convert(Vector, loss'(β)))
algorithm = LBFGS(; gradtol=1e-3, verbosity=2)
βᵒᵖᵗ, _ = optimize(loss_∇loss, β⁰, algorithm)

@show loss(βᵒᵖᵗ)
@show loss'(βᵒᵖᵗ)
