using ITensors

# TODO: this uses all of the tests to make
# precompile statements, but takes a long time
# (e.g. 700 seconds).
# Try again with later versions of PackageCompiler
#
# include(joinpath(joinpath(dirname(dirname(@__DIR__)), 
#                           test"),
#                  "runtests.jl"))

N = 10
sites = siteinds("S=1",N)
ampo = AutoMPO()
for j=1:N-1
    ampo .+= ("Sz",j,"Sz",j+1)
    ampo .+= (0.5,"S+",j,"S-",j+1)
    ampo .+= (0.5,"S-",j,"S+",j+1)
end
H = MPO(ampo,sites)
psi0 = randomMPS(sites,10)
sweeps = Sweeps(5)
maxdim!(sweeps, 10,20,100,100,200)
cutoff!(sweeps, 1E-11)
energy, psi = dmrg(H,psi0, sweeps)

sites = siteinds("S=1", N; conserveqns = true)
ampo = AutoMPO()
for j=1:N-1
    ampo .+= ("Sz",j,"Sz",j+1)
    ampo .+= (0.5,"S+",j,"S-",j+1)
    ampo .+= (0.5,"S-",j,"S+",j+1)
end
H = MPO(ampo,sites)
psi0 = randomMPS(sites,10)
sweeps = Sweeps(5)
maxdim!(sweeps, 10,20,100,100,200)
cutoff!(sweeps, 1E-11)
energy, psi = dmrg(H,psi0, sweeps)
