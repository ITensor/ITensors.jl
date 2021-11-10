using Test
using ITensors

using ITensors: ∑, ∏

@testset "Simple trotterization" begin
  H = ∑{Op}() + ("X", 1) + ("Y", 1)

  s = siteinds("Qubit", 1)

  for nsteps in [10, 100, 1000]
    expHᵉˣᵃᶜᵗ = ITensor(exp(H), s)
    @test expHᵉˣᵃᶜᵗ ≈ ITensor(exp(H; alg=Trotter{1}(nsteps)), s) rtol = 1 / nsteps
    @test expHᵉˣᵃᶜᵗ ≈ ITensor(exp(H; alg=Trotter{2}(nsteps)), s) rtol = (1 / nsteps)^2
    @test expHᵉˣᵃᶜᵗ ≈ ITensor(exp(H; alg=Trotter{4}(nsteps)), s) rtol = (1 / nsteps)^2
    @test expHᵉˣᵃᶜᵗ ≈ ITensor(exp(H; alg=Trotter{8}(nsteps)), s) rtol = (1 / nsteps)^2

    # Convert to ITensors
    t = 1.0
    Uᵉˣᵃᶜᵗ = ∏([ITensor(exp(im * t * H), s)])
    U = ∏{ITensor}(exp(im * t * H; alg=Trotter{2}(nsteps)), s)
    ψ₀ = onehot(s .=> "0")
    Uᵉˣᵃᶜᵗψ₀ = Uᵉˣᵃᶜᵗ(ψ₀)
    Uψ₀ = U(ψ₀)
    @test Uᵉˣᵃᶜᵗψ₀ ≈ Uψ₀ rtol = (1 / nsteps)^2
  end
end

function heisenberg(N)
  os = Sum{Op}()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end
  return os
end

@testset "Heisenberg Trotter" begin
  N = 4
  ℋ = heisenberg(N)
  s = siteinds("S=1/2", N)
  ψ₀ = MPS(s, n -> isodd(n) ? "↑" : "↓")
  t = 1.0
  for nsteps in [10, 100]
    for order in [1, 2, 4]
      𝒰 = exp(im * t * ℋ; alg=Trotter{order}(nsteps))
      U = ∏{ITensor}(𝒰, s)
      H = ITensor(ℋ, s)
      Uʳᵉᶠψ₀ = replaceprime(exp(im * t * H) * prod(ψ₀), 1 => 0)
      atol = max(1e-6, 1 / nsteps^order)
      @test prod(U(ψ₀)) ≈ Uʳᵉᶠψ₀ atol = atol
    end
  end
end
