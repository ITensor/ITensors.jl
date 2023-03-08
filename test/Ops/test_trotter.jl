using Test
using ITensors
using ITensors.Ops

@testset "Simple trotterization" begin
  H = Sum{Op}() + ("X", 1) + ("Y", 1)

  s = siteinds("Qubit", 1)

  for nsteps in [10, 100, 1000]
    expHᵉˣᵃᶜᵗ = ITensor(exp(H), s)
    @test expHᵉˣᵃᶜᵗ ≈ ITensor(exp(H; alg=Trotter{1}(nsteps)), s) rtol = 1 / nsteps
    @test expHᵉˣᵃᶜᵗ ≈ ITensor(exp(H; alg=Trotter{2}(nsteps)), s) rtol = (1 / nsteps)^2
    @test_broken expHᵉˣᵃᶜᵗ ≈ ITensor(exp(H; alg=Trotter{4}(nsteps)), s) rtol =
      (1 / nsteps)^2
    @test_broken expHᵉˣᵃᶜᵗ ≈ ITensor(exp(H; alg=Trotter{8}(nsteps)), s) rtol =
      (1 / nsteps)^2

    # Convert to ITensors
    t = 1.0
    Uᵉˣᵃᶜᵗ = ITensor(exp(im * t * H), s)
    U = Prod{ITensor}(exp(im * t * H; alg=Trotter{2}(nsteps)), s)
    ψ₀ = onehot(s .=> "0")
    Uᵉˣᵃᶜᵗψ₀ = Uᵉˣᵃᶜᵗ(ψ₀)
    Uψ₀ = U(ψ₀)
    @test Uᵉˣᵃᶜᵗψ₀ ≈ Uψ₀ rtol = (1 / nsteps)^2
  end
end
