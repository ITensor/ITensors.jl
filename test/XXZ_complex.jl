#########
## XXZ model with imaginary part
#########
using ITensors

function Hamiltonian(sites, Δ::Float64, J::Float64, γ::Float64, h::Float64)
  N = length(sites)
  ampo = AutoMPO()

  for j in 1:(N - 1)
    ampo += Δ, "Sz", j, "Sz", j + 1
    ampo += -(J + im * γ * (-1.0)^j), "Sx", j, "Sx", j + 1
    ampo += -(J + im * γ * (-1.0)^j), "Sy", j, "Sy", j + 1
  end
  for j in 1:N
    ampo += h * (-1.0)^j, "Sz", j
  end
  #  Convert these terms to an MPO tensor network
  return MPO(ampo, sites)
end

let
  #model parameters
  N = 4
  Δ = -0.70
  J = 1.0
  γ = 0.1
  h = 0.1

  alg = "qr_iteration"

  #dmrg parameters
  sweeps = Sweeps(1000)
  minsweeps = 5
  maxdim!(sweeps, 50, 100, 200)
  #cutoff!(sweeps, 1E-12)
  etol = 1E-12

  sites = siteinds("S=1/2", N; conserve_qns=false)

  #initial state
  state = ["Emp" for n in 1:N]
  p = N
  for i in N:-1:1
    if p > i
      #println("Doubly occupying site $i")
      state[i] = "UpDn"
      p -= 2
    elseif p > 0
      #println("Singly occupying site $i")
      state[i] = (isodd(i) ? "Up" : "Dn")
      p -= 1
    end
  end
  psi0 = randomMPS(sites, state)
  @show flux(psi0)

  H = Hamiltonian(sites, Δ, J, γ, h)

  obs = DMRGObserver(
    ["Sz"], sites; energy_tol=etol, minsweeps=minsweeps, complex_energies=true
  )

  energy, psi = dmrg(
    H, psi0, sweeps; svd_alg=alg, observer=obs, outputlevel=1, ishermitian=false
  )
  println("Final energy = $energy")
end
