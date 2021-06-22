using ITensors

# TODO: this uses all of the tests to make
# precompile statements, but takes a long time
# (e.g. 700 seconds).
# Try again with later versions of PackageCompiler
#
# include(joinpath(joinpath(dirname(dirname(@__DIR__)), 
#                           test"),
#                  "runtests.jl"))

N = 6
sweeps = Sweeps(3)
maxdim!(sweeps, 10)
cutoff!(sweeps, 1E-13)

ampo = OpSum()
for j in 1:(N - 1)
  ampo .+= "Sz", j, "Sz", j + 1
  ampo .+= 0.5, "S+", j, "S-", j + 1
  ampo .+= 0.5, "S-", j, "S+", j + 1
end

sites = siteinds("S=1", N)
H = MPO(ampo, sites)
psi0 = randomMPS(sites; linkdims=2)
dmrg(H, psi0, sweeps)

sites_qn = siteinds("S=1", N; conserve_qns=true)
if !hasqns(sites_qn[1])
  throw(ErrorException("Index does not have QNs in part of precompile script"))
end
H_qn = MPO(ampo, sites_qn)
psi0_qn = randomMPS(sites_qn, [isodd(n) ? "Up" : "Dn" for n in 1:N]; linkdims=2)
dmrg(H_qn, psi0_qn, sweeps)
