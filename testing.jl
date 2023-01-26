using Pkg
Pkg.develop("ITensors")
using ITensors, CUDA, Test, Adapt, TimerOutputs, LinearAlgebra, MKL, Random

using NDTensors
BLAS.get_config()
ComplexF64
for SType in (Float64,)
  mi, mj, mk, ml, ma = 2, 3, 4, 5, 6, 7
  i = Index(mi, "i")
  j = Index(mj, "j")
  k = Index(mk, "k")
  l = Index(ml, "l")
  a = Index(ma, "a")

  A = randomCuITensor(SType, i, j, k, l)
  U_gpu, S_gpu, V_gpu = ITensorGPU.svd(A, (j, l))
  U_cpu, S_cpu, V_cpu = svd(cpu(A), (j, l))
  #@show NDTensors.data(storage(cpu(U_gpu))) - NDTensors.data(storage(U_cpu))
  #@show NDTensors.data(storage(cpu(S_gpu))), NDTensors.data(storage(S_cpu))
  @test cpu(U_gpu) * cpu(S_gpu) * cpu(V_gpu) ≈ U_cpu * S_cpu * V_cpu
end

v = fill!(CuVector{Float16}(undef, 0), randn(Float16));
v

NDTensors.generic_zeros(NDTensors.Dense{Float16,CuVector{Float16}}, 100)
NDTensors.generic_zeros(CuVector, 20)
NDTensors.generic_zeros(NDTensors.Dense, 20)
using ITensorGPU

###########
begin
  N = 10
  sites = siteinds("S=1", N)
  H = cuMPO(MPO(heisenberg(N), sites))

  psi = randomCuMPS(sites)

  sweeps = Sweeps(3)
  @test length(sweeps) == 3
  maxdim!(sweeps, 10, 20, 40)
  mindim!(sweeps, 1, 10)
  cutoff!(sweeps, 1E-11)
  noise!(sweeps, 1E-10)
  str = split(sprint(show, sweeps), '\n')
  @test length(str) > 1
  energy, psi = dmrg(H, psi, sweeps; outputlevel=0)
  @test energy < -12.0
end

a = Array{Float32}(undef, 20)
fill!(a, 20)
b = CuVector(a)

Matrix{Float64}
Array{Float64,2}

m = CuMatrix{Float32}(undef, (20, 20))
fill!(m, 10);

@show typeof(m)
d = NDTensors.Dense(m)
