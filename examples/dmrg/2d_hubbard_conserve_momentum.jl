using ITensors

include(joinpath("..", "src", "electronk.jl"))

function main(; Nx = 6,
                Ny = 3,
                U = 4.0,
                conserve_ky = true)
  t = 1.0
  N = Nx * Ny
  sweeps = Sweeps(15)
  maxdim!(sweeps, 20, 60, 100, 100, 200, 400, 800, 2000, 3000)
  cutoff!(sweeps, 1e-6)
  noise!(sweeps, 1e-6, 1e-7, 1e-8, 0.0, 1e-7, 0.0, 1e-6, 0.0, 1e-6, 0.0)

  sites = siteinds("ElecK", N;
                   conserve_qns = true,
                   conserve_ky = conserve_ky,
                   modulus_ky = Ny)

  ampo = AutoMPO()
  # hopping in y-direction
  for x in 0:Nx-1
    for ky in 0:Ny-1
      s = x * Ny + ky + 1
      disp = -2 * t * cos((2 * π / Ny) * ky)
      if abs(disp) > 1e-12
        ampo .+= disp, "Nup", s 
        ampo .+= disp, "Ndn", s
      end
    end
  end

  # hopping in x-direction
  for x in 0:Nx-2
    for ky in 0:Ny-1
      s1 = x * Ny + ky + 1
      s2 = (x + 1) * Ny + ky + 1
      ampo .+= -t, "Cdagup", s1, "Cup", s2 
      ampo .+= -t, "Cdagup", s2, "Cup", s1
      ampo .+= -t, "Cdagdn", s1, "Cdn", s2
      ampo .+= -t, "Cdagdn", s2, "Cdn", s1
    end
  end

  # Hubbard interaction
  for x in 0:Nx-1
    for ky in 0:Ny-1
      for py in 0:Ny-1
        for qy in 0:Ny-1
          s1 = x * Ny + (ky + qy + Ny) % Ny + 1
          s2 = x * Ny + (py - qy + Ny) % Ny + 1
          s3 = x * Ny + py + 1
          s4 = x * Ny + ky + 1
          if s1 == s4 && s2 == s3
            ampo .+= (U/Ny), "Ndn", s1, "Nup", s2
          elseif s1 == s4
            ampo .+= (U/Ny), "Ndn", s1, "Cdagup", s2, "Cup", s3
          elseif s2 == s3
            ampo .+= (U/Ny), "Cdagdn", s1, "Cdn", s4, "Nup", s2
          else
            ampo .+= (U/Ny), "Cdagdn", s1, "Cdagup", s2, "Cup", s3, "Cdn", s4
          end
        end
      end
    end
  end
  H = MPO(ampo, sites)

  # Create start state
  state = Vector{String}(undef, N)
  for i in 1:N
    x = (i - 1) ÷ Ny
    y = (i - 1) % Ny
    if x % 2 == 0
      if y % 2 == 0
        state[i] = "Up"
      else
        state[i] = "Dn"
      end
    else
      if y % 2 == 0
        state[i] = "Dn"
      else
        state[i] = "Up"
      end
    end
  end

  psi0 = randomMPS(sites, state, 10)

  @show sweeps

  energy, psi = dmrg(H, psi0, sweeps)
  @show Nx, Ny
  @show t, U
  @show flux(psi)
  @show maxlinkdim(psi)
  @show energy
end

main()

