let
  # One step of dmrg (dmrg itself
  # leads to an error during precompilation)
  # Add `precompile(Tuple{typeof(dmrg),MPO,MPS,Sweeps})`
  # to precompile.jl by hand.
  N = 4
  sites = siteinds("S=1", N)
  ampo = AutoMPO()
  for j in 1:(N - 1)
    ampo .+= ("Sz", j, "Sz", j + 1)
    ampo .+= (0.5, "S+", j, "S-", j + 1)
    ampo .+= (0.5, "S-", j, "S+", j + 1)
  end
  H = MPO(ampo, sites)
  psi0 = randomMPS(sites, 10)
  sweeps = Sweeps(1)
  maxdim!(sweeps, 10)
  cutoff!(sweeps, 1E-11)
  PH = ProjMPO(H)
  which_decomp = nothing
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
  for sw in 1:nsweep(sweeps)
    # This loop causes problems for precompilation
    #for (b, ha) in sweepnext(N)
    b, ha = 1, 1
    ITensors.position!(PH, psi, b)
    phi = psi[b] * psi[b + 1]
    vals, vecs = ITensors.KrylovKit.eigsolve(
      PH,
      phi,
      1,
      eigsolve_which_eigenvalue;
      ishermitian=ishermitian,
      tol=eigsolve_tol,
      krylovdim=eigsolve_krylovdim,
      maxiter=eigsolve_maxiter,
    )
    energy, phi = vals[1], vecs[1]
    ortho = ha == 1 ? "left" : "right"
    drho = nothing
    spec = replacebond!(
      psi,
      b,
      phi;
      maxdim=maxdim(sweeps, sw),
      mindim=mindim(sweeps, sw),
      cutoff=cutoff(sweeps, sw),
      eigen_perturbation=drho,
      ortho=ortho,
      which_decomp=which_decomp,
    )
    measure!(
      obs; energy=energy, psi=psi, bond=b, sweep=sw, half_sweep=ha, spec=spec, quiet=quiet
    )
    #end
  end
end
