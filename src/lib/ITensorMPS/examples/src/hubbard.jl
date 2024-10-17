
function hubbard_1d(; N::Int, t=1.0, U=0.0)
  opsum = OpSum()
  for b in 1:(N-1)
    opsum -= t, "Cdagup", b, "Cup", b + 1
    opsum -= t, "Cdagup", b + 1, "Cup", b
    opsum -= t, "Cdagdn", b, "Cdn", b + 1
    opsum -= t, "Cdagdn", b + 1, "Cdn", b
  end
  if U ≠ 0
    for n in 1:N
      opsum += U, "Nupdn", n
    end
  end
  return opsum
end

function hubbard_2d(; Nx::Int, Ny::Int, t=1.0, U=0.0, yperiodic::Bool=true)
  N = Nx * Ny
  lattice = square_lattice(Nx, Ny; yperiodic=yperiodic)
  opsum = OpSum()
  for b in lattice
    opsum -= t, "Cdagup", b.s1, "Cup", b.s2
    opsum -= t, "Cdagup", b.s2, "Cup", b.s1
    opsum -= t, "Cdagdn", b.s1, "Cdn", b.s2
    opsum -= t, "Cdagdn", b.s2, "Cdn", b.s1
  end
  if U ≠ 0
    for n in 1:N
      opsum += U, "Nupdn", n
    end
  end
  return opsum
end

function hubbard_2d_ky(; Nx::Int, Ny::Int, t=1.0, U=0.0)
  opsum = OpSum()
  for x in 0:(Nx-1)
    for ky in 0:(Ny-1)
      s = x * Ny + ky + 1
      disp = -2 * t * cos((2 * π / Ny) * ky)
      if abs(disp) > 1e-12
        opsum += disp, "Nup", s
        opsum += disp, "Ndn", s
      end
    end
  end
  for x in 0:(Nx-2)
    for ky in 0:(Ny-1)
      s1 = x * Ny + ky + 1
      s2 = (x + 1) * Ny + ky + 1
      opsum -= t, "Cdagup", s1, "Cup", s2
      opsum -= t, "Cdagup", s2, "Cup", s1
      opsum -= t, "Cdagdn", s1, "Cdn", s2
      opsum -= t, "Cdagdn", s2, "Cdn", s1
    end
  end
  if U ≠ 0
    for x in 0:(Nx-1)
      for ky in 0:(Ny-1)
        for py in 0:(Ny-1)
          for qy in 0:(Ny-1)
            s1 = x * Ny + (ky + qy + Ny) % Ny + 1
            s2 = x * Ny + (py - qy + Ny) % Ny + 1
            s3 = x * Ny + py + 1
            s4 = x * Ny + ky + 1
            opsum += (U / Ny), "Cdagdn", s1, "Cdagup", s2, "Cup", s3, "Cdn", s4
          end
        end
      end
    end
  end
  return opsum
end

function hubbard(; Nx::Int, Ny::Int=1, t=1.0, U=0.0, yperiodic::Bool=true, ky::Bool=false)
  return opsum = if Ny == 1
    hubbard_1d(; N=Nx, t, U)
  elseif ky
    hubbard_2d_ky(; Nx, Ny, t, U)
  else
    hubbard_2d(; Nx, Ny, yperiodic, t, U)
  end
end
