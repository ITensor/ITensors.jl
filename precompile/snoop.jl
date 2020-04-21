using ITensors

for filename in (
  "tagset.jl",
  "smallstring.jl",
  "index.jl",
  "indexset.jl",
  "not.jl",
  #"itensor_dense.jl",
  #"itensor_diag.jl",
  #"contract.jl",
  #"combiner.jl",
  #"trg.jl",
  #"ctmrg.jl",
  #"iterativesolvers.jl",
  #"dmrg.jl",
  #"tag_types.jl",
  #"phys_site_types.jl",
  #"decomp.jl",
  #"lattices.jl",
  #"mps.jl",
  #"mpo.jl",
  #"autompo.jl",
  #"svd.jl",
  #"qn.jl",
  #"qnindex.jl",
  #"itensor_blocksparse.jl",
  #"itensor_diagblocksparse.jl",
  #"readwrite.jl",
  #"readme.jl",
  #"examples.jl",
  )
  include("../../test/$filename")
end

for T in (Float64, ComplexF64)
  i = Index(2)
  A = randomITensor(T, i, i')
  B = randomITensor(T, i', i'')
  C = A * B
  U, S, V = svd(A, i)
  A, B = factorize(A, i)
end

# One step of DMRG (dmrg itself
# leads to an error during precompilation)
N = 100
sites = siteinds("S=1", N)
ampo = AutoMPO()
for j=1:N-1
  ampo .+= ("Sz",j,"Sz",j+1)
  ampo .+= (0.5,"S+",j,"S-",j+1)
  ampo .+= (0.5,"S-",j,"S+",j+1)
end
H = MPO(ampo,sites)
psi0 = randomMPS(sites, 10)
sweeps = Sweeps(5)
maxdim!(sweeps, 10, 20, 100, 100, 200)
cutoff!(sweeps, 1E-11)
PH = ProjMPO(H)
which_decomp = "automatic"
obs = NoObserver()
quiet = false
eigsolve_tol = 1e-14
eigsolve_krylovdim = 3
eigsolve_maxiter = 1
eigsolve_verbosity = 0
ishermitian = true
eigsolve_which_eigenvalue = :SR
psi = copy(psi0)
N = length(psi)
ITensors.position!(PH, psi0, 1)
energy = 0.0
sw = 1
b, ha = 1, 1
ITensors.position!(PH, psi, b)
phi = psi[b] * psi[b+1]
vals, vecs = ITensors.KrylovKit.eigsolve(PH, phi, 1, eigsolve_which_eigenvalue;
                                         ishermitian = ishermitian,
                                         tol = eigsolve_tol,
                                         krylovdim = eigsolve_krylovdim,
                                         maxiter = eigsolve_maxiter)
energy, phi = vals[1], vecs[1]
ortho = ha == 1 ? "left" : "right"
drho = nothing
spec = replacebond!(psi, b, phi; maxdim = maxdim(sweeps, sw),
                                 mindim = mindim(sweeps, sw),
                                 cutoff = cutoff(sweeps, sw),
                                 eigen_perturbation = drho,
                                 ortho = ortho,
                                 which_decomp = which_decomp)
measure!(obs; energy = energy,
              psi = psi,
              bond = b,
              sweep = sw,
              half_sweep = ha,
              spec = spec,
              quiet = quiet)
