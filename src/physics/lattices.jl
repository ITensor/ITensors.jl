export LatticeBond,
       Lattice,
       squareLattice

struct LatticeBond
  s1::Int
  s2::Int
  x1::Float64
  y1::Float64
  x2::Float64
  y2::Float64
  type::String
end

function LatticeBond(s1::Int,s2::Int,
                     x1::Real,y1::Real,
                     x2::Real,y2::Real,
                     bondtype::String="")
  cf(x) = convert(Float64,x)
  return LatticeBond(s1,s2,cf(x1),cf(y1),cf(x2),cf(y2),bondtype)
end

const Lattice = Vector{LatticeBond}

function squareLattice(Nx::Int,
                       Ny::Int;
                       kwargs...)::Lattice
  yperiodic = get(kwargs,:yperiodic,false)
  yperiodic = yperiodic && (Ny > 2)
  N = Nx*Ny
  Nbond = 2N-Ny + (yperiodic ? 0 : -Nx)
  latt = Lattice(undef,Nbond)
  b = 0
  for n=1:N
    x = div(n-1,Ny)+1
    y = mod(n-1,Ny)+1
    if x < Nx
      latt[b+=1] = LatticeBond(n,n+Ny,x,y,x+1,y)
    end
    if Ny > 1
      if y < Ny
        latt[b+=1] = LatticeBond(n,n+1,x,y,x,y+1);
      end
      if yperiodic && y==1
        latt[b+=1] = LatticeBond(n,n+Ny-1,x,y,x,y+Ny)
      end
    end
  end
  return latt
end

