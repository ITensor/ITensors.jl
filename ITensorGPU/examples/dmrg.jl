using ITensors
using ITensorGPU

# Set to identity to run on CPU
gpu = cu

N = 50
sites = siteinds("S=1", N)

ampo = AutoMPO()
for j in 1:(N - 1)
  ampo .+= 0.5, "S+", j, "S-", j + 1
  ampo .+= 0.5, "S-", j, "S+", j + 1
  ampo .+= "Sz", j, "Sz", j + 1
end
H = gpu(MPO(ampo, sites))

ψ₀ = gpu(randomMPS(sites))

sweeps = Sweeps(6)
maxdim!(sweeps, 10, 20, 40, 100)
mindim!(sweeps, 1, 10)
cutoff!(sweeps, 1e-11)
noise!(sweeps, 1e-10)
energy, ψ = @time dmrg(H, ψ₀, sweeps)
@show energy
