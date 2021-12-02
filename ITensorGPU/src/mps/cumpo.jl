function cuMPO(O::MPO)
    P = copy(O)
    for site in 1:length(O)
        P.data[site] = cuITensor(O.data[site])
    end
    return P 
end
cuMPO() = MPO()
  
cuMPO(A::Vector{ITensor}) = cuMPO(MPO(A))
cuMPO(sites) = cuMPO(MPO(sites))

cu(M::MPO) = cuMPO(M)

function randomCuMPO(sites, m::Int=1)
  M = cuMPO(sites)
  for i ∈ eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  m > 1 && throw(ArgumentError("randomMPO: currently only m==1 supported"))
  return M
end

function cpu(M::T) where {T <: Union{MPS, MPO}}
    if typeof(tensor(ITensors.data(M)[1])) <: CuDenseTensor
        return T(cpu.(ITensors.data(M)), M.llim, M.rlim)    
    else
        return M
    end
end
