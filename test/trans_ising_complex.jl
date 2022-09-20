using ITensors

function Hamiltonian(sites, λ::Float64, k::Float64)
  N = length(sites)
  ampo = AutoMPO()

  for j in 1:(N - 1)
    ampo += -0.5 * 2 * λ, "Sz", j, "Sz", j + 1
  end
  for j in 1:N
    ampo += -0.5 * 2 * im * k, "Sz", j
    ampo += -0.5, "S+", j
    ampo += -0.5, "S-", j
  end
  #  Convert these terms to an MPO tensor network
  return MPO(ampo, sites)
end

let
  #-----------------------------------------------------------------------

  #models parameters
  N = 4
  λ = 0.1
  k = 0.5

  alg = "qr_iteration"

  #dmrg parameters
  sweeps = Sweeps(1000)
  minsweeps = 5
  maxdim!(sweeps, 50, 100, 200)
  #cutoff!(sweeps, 1E-12)
  etol = 1E-12

  #-----------------------------------------------------------------------

  sites = siteinds("S=1/2", N; conserve_qns=false)
  #-----------------------------------------------------------------------

  #inicial state
  state = ["Emp" for n in 1:N]
  p = N
  for i in N:-1:1
    if p > i
      state[i] = "UpDn"
      p -= 2
    elseif p > 0
      state[i] = (isodd(i) ? "Up" : "Dn")
      p -= 1
    end
  end
  psi0 = randomMPS(sites, state)
  @show flux(psi0)
  #-----------------------------------------------------------------------

  #-----------------------------------------------------------------------

  H = Hamiltonian(sites, λ, k)
  #-----------------------------------------------------------------------

  obs = DMRGObserver(; energy_tol=etol, minsweeps=2, complex_energies=true)

  energy, psi = dmrg(
    H, psi0, sweeps; svd_alg=alg, observer=obs, outputlevel=1, ishermitian=false
  )
  println("Final energy = $energy")
end
