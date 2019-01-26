
struct MPS
  N_::Int
  A_::Vector{ITensor}
  MPS() = new(0,Vector{ITensor}())

  function MPS(sites::SiteSet)
    N = length(sites)
    new(N,fill(ITensor(),N))
  end
end

length(m::MPS) = m.N_
getindex(m::MPS, n::Integer) = getindex(m.A_,n)

import Base.show
function show(io::IO,
              psi::MPS)
  print(io,"MPS")
  (length(psi) > 0) && print(io,"\n")
  for i=1:length(psi)
    println(io,"$i  $(psi[i])")
  end
end

