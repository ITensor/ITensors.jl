using Test
using ITensors
using ITensorGPU
using PastaQ
using Zygote
using OptimKit

function ising_model(n; J=1.0, h)
  H = OpSum()
  for j in 1:n
    if j < n
      H -= J, "Z", j, "Z", j + 1
    end
    H -= h, "X", j
  end
  return H
end

Ry(θ) = [("Ry", j, (θ=θ[j],)) for j in 1:length(θ)]
CNOT(n) = [("CNOT", j, j + 1) for j in 1:(n - 1)]
function U(θ)
  nlayers = length(θ)
  Uθ = Tuple[]
  for l in 1:(nlayers - 1)
    Uθ = [Uθ; [Ry(θ[l]); CNOT(length(θ[l]))]]
  end
  Uθ = [Uθ; Ry(θ[nlayers])]
  return Uθ
end

function f(θ, ψ; kwargs...)
  i = siteinds(ψ)
  ψθ = runcircuit(i, U(θ); kwargs...)
  return 1 - abs(inner(ψ, ψθ))
end

function f_∇f(f, θ, ψ; kwargs...)
  fθ, (∇fθ,) = withgradient(θ -> f(θ, ψ; kwargs...), θ)
  return fθ, ∇fθ
end

@testset "PastaQ runcircuit on $device with element type $eltype" for device in (cpu, cu),
  eltype in (Float32, Float64)

  n = 4
  h = 0.5
  cutoff = 1e-5
  gradtol = 1e-4
  maxdim = 10
  maxiter = 30
  nlayers = 4
  i = siteinds("Qubit", n)
  ψ = device(eltype, MPS(i, j -> isodd(j) ? "0" : "1"))
  H = device(eltype, MPO(ising_model(n; h), i))
  _, ψ = dmrg(H, ψ; nsweeps=10, cutoff, maxdim, outputlevel=0)
  θ = [zeros(eltype, n) for l in 1:nlayers]
  (θ,) = optimize(
    θ -> f_∇f(f, θ, ψ; cutoff, maxdim, device, eltype),
    θ,
    LBFGS(; verbosity=0, maxiter, gradtol),
  )
  ψθ = runcircuit(i, U(θ); cutoff, maxdim, device, eltype)
  energy_reference = inner(ψ', H, ψ)
  energy_opt = inner(ψθ', H, ψθ)
  is_device(x, device) = device == cu ? ITensorGPU.is_cu(x) : !ITensorGPU.is_cu(x)
  @test is_device(H, device)
  @test is_device(ψ, device)
  @test is_device(ψθ, device)
  @test ITensors.scalartype(H) <: eltype
  @test ITensors.scalartype(ψ) <: eltype
  @test ITensors.scalartype(ψθ) <: eltype
  @test inner(ψ', H, ψ) ≈ inner(ψθ', H, ψθ) atol = 1e-3
end
