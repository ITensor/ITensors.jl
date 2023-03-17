using ITensors
using OptimKit
using Random
using Test
using Zygote

include(joinpath(@__DIR__, "utils", "circuit.jl"))

@testset "optimization" begin
  @testset "Energy minimization" begin
    N = 3
    s = siteinds("S=1/2", N; conserve_qns=true)
    os = OpSum()
    for n in 1:(N - 1)
      os .+= 0.5, "S+", n, "S-", n + 1
      os .+= 0.5, "S-", n, "S+", n + 1
      os .+= "Sz", n, "Sz", n + 1
    end
    Hmpo = MPO(os, s)
    ψ₀mps = randomMPS(s, n -> isodd(n) ? "↑" : "↓")
    H = prod(Hmpo)
    ψ₀ = prod(ψ₀mps)
    # The Rayleigh quotient to minimize
    function E(H::ITensor, ψ::ITensor)
      ψdag = dag(ψ)
      return (ψdag' * H * ψ)[] / (ψdag * ψ)[]
    end
    E(ψ::ITensor) = E(H, ψ)
    ∇E(ψ::ITensor) = E'(ψ)
    fg(ψ::ITensor) = (E(ψ), ∇E(ψ))
    linesearch = HagerZhangLineSearch(;
      c₁=0.1, c₂=0.9, ϵ=1e-6, θ=1 / 2, γ=2 / 3, ρ=5.0, verbosity=0
    )
    algorithm = LBFGS(3; maxiter=20, gradtol=1e-8, linesearch=linesearch)
    ψ, fψ, gψ, numfg, normgradhistory = optimize(fg, ψ₀, algorithm)
    D, _ = eigen(H; ishermitian=true)
    @test E(H, ψ) < E(H, ψ₀)
    @test E(H, ψ) ≈ minimum(D)
  end

  @testset "Energy minimization (MPS)" begin
    N = 4
    χ = 4
    s = siteinds("S=1/2", N; conserve_qns=true)
    os = OpSum()
    for n in 1:(N - 1)
      os .+= 0.5, "S+", n, "S-", n + 1
      os .+= 0.5, "S-", n, "S+", n + 1
      os .+= "Sz", n, "Sz", n + 1
    end
    Hmpo = MPO(os, s)

    Random.seed!(1234)
    ψ₀mps = randomMPS(s, n -> isodd(n) ? "↑" : "↓"; linkdims=χ)

    H = ITensors.data(Hmpo)
    ψ₀ = ITensors.data(ψ₀mps)
    # The Rayleigh quotient to minimize
    function E(H::Vector{ITensor}, ψ::Vector{ITensor})
      N = length(ψ)
      ψdag = dag.(addtags.(ψ, "bra"; tags="Link"))
      ψ′dag = prime.(ψdag)
      e = ITensor(1.0)
      for n in 1:N
        e = e * ψ′dag[n] * H[n] * ψ[n]
      end
      norm = ITensor(1.0)
      for n in 1:N
        norm = norm * ψdag[n] * ψ[n]
      end
      return e[] / norm[]
    end
    E(ψ) = E(H, ψ)
    ∇E(ψ) = E'(ψ)
    fg(ψ) = (E(ψ), ∇E(ψ))
    linesearch = HagerZhangLineSearch(; c₁=0.1, c₂=0.9, ϵ=1e-6, θ=1 / 2, γ=2 / 3, ρ=5.0)
    algorithm = LBFGS(5; maxiter=50, gradtol=1e-4, linesearch=linesearch, verbosity=0)
    ψ, fψ, gψ, numfg, normgradhistory = optimize(fg, ψ₀, algorithm)
    sweeps = Sweeps(5)
    setmaxdim!(sweeps, χ)
    fψmps, ψmps = dmrg(Hmpo, ψ₀mps, sweeps; outputlevel=0)
    @test E(H, ψ) ≈ inner(ψmps', Hmpo, ψmps) / inner(ψmps, ψmps) rtol = 1e-2
  end

  @testset "State preparation (full state)" begin
    function Rylayer(N, θ⃗)
      return [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
    end

    function CXlayer(N)
      return [("CX", (n, n + 1)) for n in 1:2:(N - 1)]
    end

    # The variational circuit we want to optimize
    function variational_circuit(θ⃗)
      N = length(θ⃗)
      return vcat(Rylayer(N, θ⃗), CXlayer(N))
    end

    N = 4
    Random.seed!(1234)
    θ⃗ = 2π .* rand(N)
    gates = variational_circuit(θ⃗)

    s = siteinds("Qubit", N)
    ψₘₚₛ = MPS(s, "0")
    ψ = prod(ψₘₚₛ)
    U = buildcircuit(gates, s)
    # Create the target state
    Uψ = apply(U, ψ)

    @test inner_circuit(Uψ, U, ψ) ≈ 1

    function loss(θ⃗)
      gates = variational_circuit(θ⃗)
      U = buildcircuit(gates, s)
      return -abs(inner_circuit(Uψ, U, ψ))^2
    end

    θ⃗₀ = randn!(copy(θ⃗))
    fg(x) = (loss(x), convert(Vector, loss'(x)))
    θ⃗ₒₚₜ, fₒₚₜ, gₒₚₜ, numfg, normgradhistory = optimize(fg, θ⃗₀, GradientDescent())
    @test loss(θ⃗ₒₚₜ) ≈ loss(θ⃗) rtol = 1e-2
  end

  @testset "State preparation (MPS)" begin
    for gate in ["Ry"] #="Rx", =#
      nsites = 4 # Number of sites
      nlayers = 2 # Layers of gates in the ansatz
      gradtol = 1e-3 # Tolerance for stopping gradient descent

      # A layer of the circuit we want to optimize
      function layer(nsites, θ⃗)
        gate_layer = [(gate, (n,), (θ=θ⃗[n],)) for n in 1:nsites]
        CX_layer = [("CX", (n, n + 1)) for n in 1:2:(nsites - 1)]
        return [gate_layer; CX_layer]
      end

      # The variational circuit we want to optimize
      function variational_circuit(nsites, nlayers, θ⃗)
        range = 1:nsites
        circuit = layer(nsites, θ⃗[range])
        for n in 1:(nlayers - 1)
          circuit = [circuit; layer(nsites, θ⃗[range .+ n * nsites])]
        end
        return circuit
      end

      Random.seed!(1234)

      θ⃗ᵗᵃʳᵍᵉᵗ = 2π * rand(nsites * nlayers)
      𝒰ᵗᵃʳᵍᵉᵗ = variational_circuit(nsites, nlayers, θ⃗ᵗᵃʳᵍᵉᵗ)

      s = siteinds("Qubit", nsites)
      Uᵗᵃʳᵍᵉᵗ = ops(𝒰ᵗᵃʳᵍᵉᵗ, s)

      ψ0 = MPS(s, "0")

      # Create the random target state
      ψᵗᵃʳᵍᵉᵗ = apply(Uᵗᵃʳᵍᵉᵗ, ψ0; cutoff=1e-8)

      #
      # The loss function, a function of the gate parameters
      # and implicitly depending on the target state:
      #
      # loss(θ⃗) = -|⟨θ⃗ᵗᵃʳᵍᵉᵗ|U(θ⃗)|0⟩|² = -|⟨θ⃗ᵗᵃʳᵍᵉᵗ|θ⃗⟩|²
      #
      function loss(θ⃗)
        nsites = length(ψ0)
        s = siteinds(ψ0)
        𝒰θ⃗ = variational_circuit(nsites, nlayers, θ⃗)
        Uθ⃗ = ops(𝒰θ⃗, s)
        ψθ⃗ = apply(Uθ⃗, ψ0; cutoff=1e-8)
        return -abs(inner(ψᵗᵃʳᵍᵉᵗ, ψθ⃗))^2
      end

      θ⃗₀ = randn!(copy(θ⃗ᵗᵃʳᵍᵉᵗ))

      @test loss(θ⃗₀) ≉ loss(θ⃗ᵗᵃʳᵍᵉᵗ)

      loss_∇loss(x) = (loss(x), convert(Vector, loss'(x)))
      @show gate
      algorithm = LBFGS(; gradtol=gradtol, verbosity=2)
      θ⃗ₒₚₜ, lossₒₚₜ, ∇lossₒₚₜ, numfg, normgradhistory = optimize(
        loss_∇loss, θ⃗₀, algorithm
      )

      @test loss(θ⃗ₒₚₜ) ≈ loss(θ⃗ᵗᵃʳᵍᵉᵗ) rtol = 1e-5
    end
  end

  @testset "VQE (MPS)" begin
    nsites = 4 # Number of sites
    nlayers = 2 # Layers of gates in the ansatz
    gradtol = 1e-3 # Tolerance for stopping gradient descent

    # The Hamiltonian we are minimizing
    function ising_hamiltonian(nsites; h)
      ℋ = OpSum()
      for j in 1:(nsites - 1)
        ℋ += -1, "Z", j, "Z", j + 1
      end
      for j in 1:nsites
        ℋ += h, "X", j
      end
      return ℋ
    end

    # A layer of the circuit we want to optimize
    function layer(nsites, θ⃗)
      RY_layer = [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:nsites]
      CX_layer = [("CX", (n, n + 1)) for n in 1:2:(nsites - 1)]
      return [RY_layer; CX_layer]
    end

    # The variational circuit we want to optimize
    function variational_circuit(nsites, nlayers, θ⃗)
      range = 1:nsites
      circuit = layer(nsites, θ⃗[range])
      for n in 1:(nlayers - 1)
        circuit = [circuit; layer(nsites, θ⃗[range .+ n * nsites])]
      end
      return circuit
    end

    s = siteinds("Qubit", nsites)

    h = 1.3
    ℋ = ising_hamiltonian(nsites; h=h)
    H = MPO(ℋ, s)
    ψ0 = MPS(s, "0")

    #
    # The loss function, a function of the gate parameters
    # and implicitly depending on the Hamiltonian and state:
    #
    # loss(θ⃗) = ⟨0|U(θ⃗)† H U(θ⃗)|0⟩ = ⟨θ⃗|H|θ⃗⟩
    #
    function loss(θ⃗)
      nsites = length(ψ0)
      s = siteinds(ψ0)
      𝒰θ⃗ = variational_circuit(nsites, nlayers, θ⃗)
      Uθ⃗ = ops(𝒰θ⃗, s)
      ψθ⃗ = apply(Uθ⃗, ψ0; cutoff=1e-8)
      return inner(ψθ⃗', H, ψθ⃗; cutoff=1e-8)
    end

    Random.seed!(1234)
    θ⃗₀ = 2π * rand(nsites * nlayers)

    loss_∇loss(x) = (loss(x), convert(Vector, loss'(x)))
    algorithm = LBFGS(; gradtol=gradtol, verbosity=0)
    θ⃗ₒₚₜ, lossₒₚₜ, ∇lossₒₚₜ, numfg, normgradhistory = optimize(loss_∇loss, θ⃗₀, algorithm)

    sweeps = Sweeps(5)
    setmaxdim!(sweeps, 10)
    e_dmrg, ψ_dmrg = dmrg(H, ψ0, sweeps; outputlevel=0)

    @test loss(θ⃗ₒₚₜ) ≈ e_dmrg rtol = 1e-1
  end
end
