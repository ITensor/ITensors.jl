using ITensors.ITensorMPS: MPO, OpSum, dmrg, random_mps, siteinds

# TODO: This uses all of the tests to make
# precompile statements, but takes a long time
# (e.g. 700 seconds).
# Try again with later versions of PackageCompiler
#
# include(joinpath(joinpath(dirname(dirname(@__DIR__)),
#                           test"),
#                  "runtests.jl"))

function main(; N, dmrg_kwargs)
  opsum = OpSum()
  for j in 1:(N-1)
    opsum += 0.5, "S+", j, "S-", j + 1
    opsum += 0.5, "S-", j, "S+", j + 1
    opsum += "Sz", j, "Sz", j + 1
  end
  for conserve_qns in (false, true)
    sites = siteinds("S=1", N; conserve_qns)
    H = MPO(opsum, sites)
    ψ0 = random_mps(sites, j -> isodd(j) ? "↑" : "↓"; linkdims=2)
    dmrg(H, ψ0; outputlevel=0, dmrg_kwargs...)
  end
  return nothing
end

main(; N=6, dmrg_kwargs=(; nsweeps=3, maxdim=10, cutoff=1e-13))
