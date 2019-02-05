function cuMPO(O::MPO)
    P = copy(O)
    for site in 1:length(O)
        P.A_[site] = cuITensor(O.A_[site])
    end
    return P 
end
  
cuMPO(N::Int, A::Vector{ITensor}) = cuMPO(MPO(N, A))
cuMPO(sites::SiteSet) = cuMPO(MPO(sites))

function cuMPO(::Type{T}, 
               sites::SiteSet, 
               ops::Vector{String}) where {T}
    return cuMPO(MPO(T, sites, ops))
end

function cuMPO(::Type{T}, 
               sites::SiteSet, 
               ops::String;
               store_type::DataType = Float64) where {T}
    return cuMPO(MPO(T, sites, fill(ops, length(sites)), store_type=store_type))
end
