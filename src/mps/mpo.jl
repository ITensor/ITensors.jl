
struct MPO
  N_::Int
  A_::Vector{ITensor}

  MPO() = new(0,Vector{ITensor}())

  function MPO(N::Int, A::Vector{ITensor})
    new(N,A)
  end
  
  function MPO(sites::SiteSet)
    N = length(sites)
    new(N,fill(ITensor(),N))
  end
end

length(m::MPO) = m.N_

getindex(m::MPO, n::Integer) = getindex(m.A_,n)
setindex!(m::MPO,T::ITensor,n::Integer) = setindex!(m.A_,T,n)

copy(m::MPO) = MPO(m.N_,copy(m.A_))

function show(io::IO,
              W::MPO)
  print(io,"MPO")
  (length(W) > 0) && print(io,"\n")
  for i=1:length(W)
    println(io,"$i  $(W[i])")
  end
end
