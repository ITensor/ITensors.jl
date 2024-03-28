using ITensors
using ITensorGPU

# Set to identity to run on CPU
device = cu

N = 50
sites = siteinds("S=1", N)

opsum = OpSum()
for j in 1:(N - 1)
  opsum .+= 0.5, "S+", j, "S-", j + 1
  opsum .+= 0.5, "S-", j, "S+", j + 1
  opsum .+= "Sz", j, "Sz", j + 1
end
H = device(MPO(opsum, sites))

ψ₀ = device(randomMPS(sites))

dmrg_kwargs = (;
  nsweeps=6, maxdim=[10, 20, 40, 100], mindim=[1, 10], cutoff=1e-11, noise=1e-10
)
energy, ψ = @time dmrg(H, ψ₀; dmrg_kwargs...)
@show energy
