module BenchDMRG

using BenchmarkTools
using ITensors

suite = BenchmarkGroup()

let N = 100
  sites = siteinds("S=1", N)
  ampo = OpSum()
  for j in 1:(N - 1)
    ampo .+= ("Sz", j, "Sz", j + 1)
    ampo .+= (0.5, "S+", j, "S-", j + 1)
    ampo .+= (0.5, "S-", j, "S+", j + 1)
  end
  H = MPO(ampo, sites)
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  psi = productMPS(sites, state)
  sweeps = Sweeps(5)
  maxdim!(sweeps, 10, 20, 100, 100, 200)
  cutoff!(sweeps, 1e-11)

  # Precompile
  dmrg(H, psi, sweeps; outputlevel=0)

  suite["1d_S=1_heisenberg"] = @benchmarkable dmrg($H, $psi, $sweeps; outputlevel=0)
end

let N = 100
  sites = siteinds("S=1", N; conserve_qns=true)
  ampo = OpSum()
  for j in 1:(N - 1)
    ampo .+= ("Sz", j, "Sz", j + 1)
    ampo .+= (0.5, "S+", j, "S-", j + 1)
    ampo .+= (0.5, "S-", j, "S+", j + 1)
  end
  H = MPO(ampo, sites)
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  psi = productMPS(sites, state)
  sweeps = Sweeps(5)
  maxdim!(sweeps, 10, 20, 100, 100, 200)
  cutoff!(sweeps, 1E-10)

  # Precompile
  dmrg(H, psi, sweeps; outputlevel=0)

  suite["1d_S=1_heisenberg_qn"] = @benchmarkable dmrg($H, $psi, $sweeps; outputlevel=0)
end

#let Ny = 6,
#    Nx = 12
#  N = Nx * Ny
#  sites = siteinds("S=1/2", N)
#  lattice = square_lattice(Nx, Ny; yperiodic = false)
#  ampo = OpSum()
#  for b in lattice
#    ampo .+= (0.5, "S+", b.s1, "S-", b.s2)
#    ampo .+= (0.5, "S-", b.s1, "S+", b.s2)
#    ampo .+= (     "Sz", b.s1, "Sz", b.s2)
#  end
#  H = MPO(ampo, sites)
#  state = [isodd(n) ? "Up" : "Dn" for n=1:N]
#  psi = productMPS(sites, state)
#  sweeps = Sweeps(10)
#  maxdim!(sweeps, 10, 20, 100, 100, 200, 400, 800)
#  cutoff!(sweeps, 1e-8)
#
#  # Precompile
#  dmrg(H, psi, sweeps; outputlevel = 0)
#
#  suite["2d_S=1/2_heisenberg"] = @benchmarkable dmrg($H, $psi, $sweeps;
#                                                   outputlevel = 0)
#end

end

BenchDMRG.suite
