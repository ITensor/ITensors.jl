
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

  function MPO(sites::SiteSet, 
               ops::Vector{String})
    N = length(sites)
    its = Vector{ITensor}(undef, N)
    links = Vector{Index}(undef, N)
    for ii in 1:N
        si = sites[ii]
        spin_op = op(sites, ops[ii])
        links[ii] = Index(1, "Link,n=$ii")
        local this_it
        if ii == 1
            this_it = ITensor(links[ii], si, si')
            this_it[links[ii](1), s[:], s'[:]] = spin_op[si[:], si'[:]]
        elseif ii == N
            this_it = ITensor(links[ii-1], si, si')
            this_it[links[ii-1](1), si[:], si'[:]] = spin_op[si[:], si'[:]]
        else
            this_it = ITensor(links[ii-1], links[ii], si, si')
            this_it[links[ii-1](1), links[ii](1), si[:], si'[:]] = spin_op[si[:], si'[:]]
        end
        its[ii] = this_it
    end
    new(N,its)
  end

  function MPO(sites::SiteSet, 
               ops::String)
    return MPO(sites, fill(ops, length(sites)))
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
