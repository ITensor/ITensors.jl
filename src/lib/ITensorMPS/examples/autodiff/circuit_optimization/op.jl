using ITensors
using Zygote

s = siteind("Qubit")

f(x) = op("Ry", s; θ=x)[1, 1]

x = 0.2
@show f(x), cos(x / 2)
@show f'(x), -sin(x / 2) / 2

# Simple gate optimization
ψ0 = state(s, "0")
ψp = state(s, "+")

function loss(x)
  U = op("Ry", s; θ=x)
  Uψ0 = replaceprime(U * ψ0, 1 => 0)
  return -(dag(ψp) * Uψ0)[]
end

# Extremely simple gradient descent implementation,
# where gradients are computing with automatic differentiation
# using Zygote.
function gradient_descent(f, x0; γ, nsteps, grad_tol)
  @show γ, nsteps
  x = x0
  f_x = f(x)
  ∇f_x = f'(x)
  step = 0
  @show step, x, f_x, ∇f_x
  for step in 1:nsteps
    x -= γ * ∇f_x
    f_x = f(x)
    ∇f_x = f'(x)
    @show step, x, f_x, ∇f_x
    if norm(∇f_x) ≤ grad_tol
      break
    end
  end
  return x, f_x, ∇f_x
end

x0 = 0
γ = 2.0 # Learning rate
nsteps = 30 # Number of steps of gradient descent
grad_tol = 1e-4 # Stop if gradient falls below this value
x, loss_x, ∇loss_x = gradient_descent(loss, x0; γ=γ, nsteps=nsteps, grad_tol=grad_tol)

@show x0, loss(x0)
@show x, loss(x)
@show π / 2, loss(π / 2)
