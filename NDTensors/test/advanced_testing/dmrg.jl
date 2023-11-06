using ITensors
using NDTensors
using Test
if "cuda" in ARGS || "all" in ARGS
  using CUDA
end
if "metal" in ARGS || "all" in ARGS
  using Metal
end

function test_dmrg(N::Integer, dev::Function, cut::Float64, ref_energy, ref_time)
  # Create N spin-one degrees of freedom
  sites = siteinds("S=1", N)

  # Input operator terms which define a Hamiltonian
  os = OpSum()
  for j in 1:(N - 1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end

  # Convert these terms to an MPO tensor network
  H = dev(MPO(os, sites))

  # Create an initial random matrix product state
  psi0 = dev(randomMPS(sites; linkdims=10))

  # Plan to do 5 DMRG sweeps:
  nsweeps = 5
  # Set maximum MPS bond dimensions for each sweep
  maxdim = [10, 20, 100, 100, 200]
  # Set maximum truncation error allowed when adapting bond dimensions
  cutoff = [cut]

  # Run the DMRG algorithm, returning energy and optimized MPS
  energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel=0)
  @test energy ≈ ref_energy

  @elapsed dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel=0)
  tg = @elapsed dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel=0)
  @test tg < ref_time
end

include("../device_list.jl")
devs = devices_list(ARGS)

ref_energies = Vector{Float64}([
  -2.0000000000000004, -2.0000000000000004, -10.12463722168637, -10.124637222358869
])
ref_cpu_times = Vector{Float64}([0.005, 0.005, 2.2, 2.2])
## actual time
## Vector{Float64}([ 0.003366999,0.003374851,0.226856982,0.184108744])
## When running test NDTensors last two timings are ~2.1s
ref_cuda_times = Vector{Float64}([0.05, 0.05, 1.5, 1.5])
## actual time
## Vector{Float64}([tg = 0.020535662,0.022319508,0.512673502, 0.42965219])
## When running test(NDTensors; test_args=["cuda"]) last two timings are ~1.4s
@testset "Testing DMRG different backends" for dev in devs
  count = 0
  ref_times = if dev == NDTensors.cpu
    ref_cpu_times
  elseif dev == NDTensors.cu
    ref_cuda_times
  elseif dev == NDTensors.mtl
    nothing
  end
  for n in [2, 8], cut in [1e-10, 0.0]
    test_dmrg(n, dev, cut, ref_energies[count % 4 + 1], ref_times[count % 4 + 1])
    count += 1
  end
end
