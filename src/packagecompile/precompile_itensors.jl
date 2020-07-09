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
sites = siteinds("S=1",N)
ampo = AutoMPO()
for j=1:N-1
    ampo .+= "Sz",j,"Sz",j+1
    ampo .+= 0.5,"S+",j,"S-",j+1
    ampo .+= 0.5,"S-",j,"S+",j+1
end
H = MPO(ampo,sites)
psi0 = randomMPS(sites,2)
sweeps = Sweeps(1)
maxdim!(sweeps, 10)
cutoff!(sweeps, 1E-13)
energy, psi = dmrg(H,psi0, sweeps)

sites = siteinds("S=1", N; conserve_qns=true)
if !hasqns(sites[1])
  throw(ErrorException("Index does not have QNs in part of precompile script"))
end
ampo = AutoMPO()
for j=1:N-1
    ampo .+= ("Sz",j,"Sz",j+1)
    ampo .+= (0.5,"S+",j,"S-",j+1)
    ampo .+= (0.5,"S-",j,"S+",j+1)
end
H = MPO(ampo,sites)
psi0 = randomMPS(sites,[isodd(n) ? "Up" : "Dn" for n=1:N],2)
sweeps = Sweeps(1)
maxdim!(sweeps, 10)
cutoff!(sweeps, 1E-13)
energy, psi = dmrg(H,psi0, sweeps)
