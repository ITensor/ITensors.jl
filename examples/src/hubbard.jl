
function hubbard_1d(; N::Int, t=1.0, U=0.0)
  ampo = OpSum()
  for b in 1:(N - 1)
    ampo .+= -t, "Cdagup", b, "Cup", b + 1
    ampo .+= -t, "Cdagup", b + 1, "Cup", b
    ampo .+= -t, "Cdagdn", b, "Cdn", b + 1
    ampo .+= -t, "Cdagdn", b + 1, "Cdn", b
  end
  if U ≠ 0
    for n in 1:N
      ampo .+= U, "Nupdn", n
    end
  end
  return ampo
end

function hubbard_2d(; Nx::Int, Ny::Int, t=1.0, U=0.0, yperiodic::Bool=true)
  N = Nx * Ny
  lattice = square_lattice(Nx, Ny; yperiodic=yperiodic)
  ampo = OpSum()
  for b in lattice
    ampo .+= -t, "Cdagup", b.s1, "Cup", b.s2
    ampo .+= -t, "Cdagup", b.s2, "Cup", b.s1
    ampo .+= -t, "Cdagdn", b.s1, "Cdn", b.s2
    ampo .+= -t, "Cdagdn", b.s2, "Cdn", b.s1
  end
  if U ≠ 0
    for n in 1:N
      ampo .+= U, "Nupdn", n
    end
  end
  return ampo
end

function hubbard_2d_ky(; Nx::Int, Ny::Int, t=1.0, U=0.0)
  ampo = OpSum()
  for x in 0:(Nx - 1)
    for ky in 0:(Ny - 1)
      s = x * Ny + ky + 1
      disp = -2 * t * cos((2 * π / Ny) * ky)
      if abs(disp) > 1e-12
        ampo .+= disp, "Nup", s
        ampo .+= disp, "Ndn", s
      end
    end
  end
  for x in 0:(Nx - 2)
    for ky in 0:(Ny - 1)
      s1 = x * Ny + ky + 1
      s2 = (x + 1) * Ny + ky + 1
      ampo .+= -t, "Cdagup", s1, "Cup", s2
      ampo .+= -t, "Cdagup", s2, "Cup", s1
      ampo .+= -t, "Cdagdn", s1, "Cdn", s2
      ampo .+= -t, "Cdagdn", s2, "Cdn", s1
    end
  end
  if U ≠ 0
    for x in 0:(Nx - 1)
      for ky in 0:(Ny - 1)
        for py in 0:(Ny - 1)
          for qy in 0:(Ny - 1)
            s1 = x * Ny + (ky + qy + Ny) % Ny + 1
            s2 = x * Ny + (py - qy + Ny) % Ny + 1
            s3 = x * Ny + py + 1
            s4 = x * Ny + ky + 1
            ampo .+= (U / Ny), "Cdagdn", s1, "Cdagup", s2, "Cup", s3, "Cdn", s4
          end
        end
      end
    end
  end
  return ampo
end

function hubbard(; Nx::Int, Ny::Int=1, t=1.0, U=0.0, yperiodic::Bool=true, ky::Bool=false)
  if Ny == 1
    ampo = hubbard_1d(; N=Nx, t=t, U=U)
  elseif ky
    ampo = hubbard_2d_ky(; Nx=Nx, Ny=Ny, t=t, U=U)
  else
    ampo = hubbard_2d(; Nx=Nx, Ny=Ny, yperiodic=yperiodic, t=t, U=U)
  end
  return ampo
end
