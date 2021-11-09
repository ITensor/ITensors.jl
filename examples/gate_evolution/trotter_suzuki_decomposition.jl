using ITensors

function heisenberg(N)
  os = Sum{Op}()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end
  return os
end

function main(N; nsteps, order)
  ℋ = heisenberg(N)
  s = siteinds("S=1/2", N)
  ψ₀ = MPS(s, n -> isodd(n) ? "↑" : "↓")
  t = 1.0
  𝒰 = exp(im * t * ℋ; alg=Trotter{order}(nsteps))
  U = Prod{ITensor}(𝒰, s)
  H = ITensor(ℋ, s)
  𝒰ʳᵉᶠ = exp(im * t * ℋ)
  Uʳᵉᶠ = ITensor(𝒰ʳᵉᶠ, s)
  Uʳᵉᶠψ₀ = replaceprime(Uʳᵉᶠ * prod(ψ₀), 1 => 0)
  return norm(prod(U(ψ₀)) - Uʳᵉᶠψ₀)
end

@show main(4; nsteps=10, order=1)
@show main(4; nsteps=10, order=2)
@show main(4; nsteps=10, order=4)
@show main(4; nsteps=100, order=1)
@show main(4; nsteps=100, order=2)
@show main(4; nsteps=100, order=4)
