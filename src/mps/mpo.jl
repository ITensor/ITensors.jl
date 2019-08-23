export MPO,
       randomMPO,
       applyMPO,
       nmultMPO,
       maxLinkDim

mutable struct MPO
  N_::Int
  A_::Vector{ITensor}
  llim_::Int
  rlim_::Int

  MPO() = new(0,Vector{ITensor}(), 0, 0)

  function MPO(N::Int, A::Vector{ITensor}, llim::Int=0, rlim::Int=N+1)
    new(N, A, llim, rlim)
  end

  function MPO(sites::SiteSet)
    N = length(sites)
    v = Vector{ITensor}(undef, N)
    l = [Index(1, "Link,l=$ii") for ii ∈ 1:N-1]
    @inbounds for ii ∈ eachindex(sites)
      s = sites[ii]
      sp = prime(s)
      if ii == 1
        v[ii] = ITensor(s, sp, l[ii])
      elseif ii == N
        v[ii] = ITensor(l[ii-1], s, sp)
      else
        v[ii] = ITensor(l[ii-1], s, sp, l[ii])
      end
    end
    new(N,v,0,N+1)
  end
 
  function MPO(sites::SiteSet, 
               ops::Vector{String})
    N = length(sites)
    its = Vector{ITensor}(undef, N)
    links = Vector{Index}(undef, N)
    @inbounds for ii ∈ eachindex(sites)
        si = sites[ii]
        spin_op = op(sites, ops[ii], ii)
        links[ii] = Index(1, "Link,n=$ii")
        local this_it
        if ii == 1
            this_it = ITensor(links[ii], si, si')
            this_it[links[ii](1), si[:], si'[:]] = spin_op[si[:], si'[:]]
        elseif ii == N
            this_it = ITensor(links[ii-1], si, si')
            this_it[links[ii-1](1), si[:], si'[:]] = spin_op[si[:], si'[:]]
        else
            this_it = ITensor(links[ii-1], links[ii], si, si')
            this_it[links[ii-1](1), links[ii](1), si[:], si'[:]] = spin_op[si[:], si'[:]]
        end
        its[ii] = this_it
    end
    new(N,its,0,N+1)
  end

  MPO(sites::SiteSet, ops::String) = MPO(sites, fill(ops, length(sites)))
end
MPO(N::Int) = MPO(N,Vector{ITensor}(undef,N))

function randomMPO(sites, m::Int=1)
  M = MPO(sites)
  @inbounds for i ∈ eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  m > 1 && throw(ArgumentError("randomMPO: currently only m==1 supported"))
  return M
end

length(m::MPO) = m.N_
tensors(m::MPO) = m.A_
leftLim(m::MPO) = m.llim_
rightLim(m::MPO) = m.rlim_

getindex(m::MPO, n::Integer) = getindex(tensors(m), n)
setindex!(m::MPO, T::ITensor, n::Integer) = setindex!(tensors(m), T, n)

copy(m::MPO) = MPO(m.N_, copy(tensors(m)))
similar(m::MPO) = MPO(m.N_, similar(tensors(m)), 0, m.N_)

function deepcopy(m::T) where {T <: Union{MPO,MPS}}
    res = similar(m)
    # otherwise we will end up modifying the elements of A!
    res.A_ = deepcopy(tensors(m))
    return res
end

eachindex(m::MPO) = 1:length(m)

# TODO: optimize finding the index a little bit
# First do: scom = commonindex(A[j],x[j])
# Then do: uniqueindex(A[j],A[j-1],A[j+1],(scom,))
function siteindex(A::MPO,x::MPS,j::Integer)
  N = length(A)
  if j == 1
    si = uniqueindex(A[j],(A[j+1],x[j]))
  elseif j == N
    si = uniqueindex(A[j],(A[j-1],x[j]))
  else
    si = uniqueindex(A[j],(A[j-1],A[j+1],x[j]))
  end
  return si
end

function siteinds(A::MPO,x::MPS)
  is = IndexSet(length(A))
  @inbounds for j in eachindex(A)
    is[j] = siteindex(A,x,j)
  end
  return is
end

"""
    dag(m::MPS)
    dag(m::MPO)

Hermitian conjugation of a matrix product state or operator `m`.
"""

function dag(m::T) where {T <: Union{MPS, MPO}}
  N = length(m)
  mdag = T(N)
  @inbounds for i ∈ eachindex(m)
    mdag[i] = dag(m[i])
  end
  return mdag
end

function prime!(M::T,vargs...) where {T <: Union{MPS,MPO}}
  @inbounds for i ∈ eachindex(M)
    prime!(M[i],vargs...)
  end
end

function primelinks!(M::T, plinc::Integer = 1) where {T <: Union{MPS,MPO}}
  @inbounds for i ∈ eachindex(M)[1:end-1]
    l = linkindex(M,i)
    prime!(M[i],plinc,l)
    prime!(M[i+1],plinc,l)
  end
end

function simlinks!(M::T) where {T <: Union{MPS,MPO}}
  @inbounds for i ∈ eachindex(M)[1:end-1]
    l = linkindex(M,i)
    l̃ = sim(l)
    #M[i] *= δ(l,l̃)
    replaceindex!(M[i],l,l̃)
    #M[i+1] *= δ(l,l̃)
    replaceindex!(M[i+1],l,l̃)
  end
end

"""
maxLinkDim(M::MPS)
maxLinkDim(M::MPO)

Get the maximum link dimension of the MPS or MPO.
"""
function maxLinkDim(M::T) where {T <: Union{MPS,MPO}}
  md = 0
  for b ∈ eachindex(M)[1:end-1] 
    md = max(md,dim(linkindex(M,b)))
  end
  md
end

function show(io::IO, W::MPO)
  print(io,"MPO")
  (length(W) > 0) && print(io,"\n")
  @inbounds for (i, A) ∈ enumerate(tensors(W))
    println(io,"$i  $(inds(A))")
  end
end

function linkindex(M::MPO,j::Integer) 
  N = length(M)
  j ≥ length(M) && error("No link index to the right of site $j (length of MPO is $N)")
  li = commonindex(M[j],M[j+1])
  if isdefault(li)
    error("linkindex: no MPO link index at link $j")
  end
  return li
end

function position!(M::MPO,
                   j::Integer)
  N = length(M)
  while leftLim(M) < (j-1)
    ll = leftLim(M)+1
    s = findinds(M[ll],"Site")
    if ll == 1
      (Q,R) = qr(M[ll],s)
    else
      li = linkindex(M,ll-1)
      (Q,R) = qr(M[ll],s,li)
    end
    M[ll] = Q
    M[ll+1] *= R
    M.llim_ += 1
  end

  while rightLim(M) > (j+1)
    rl = rightLim(M)-1
    s = findinds(M[rl],"Site")
    if rl == N
      (Q,R) = qr(M[rl],s)
    else
      ri = linkindex(M,rl)
      (Q,R) = qr(M[rl],s,ri)
    end
    M[rl] = Q
    M[rl-1] *= R
    M.rlim_ -= 1
  end
  M.llim_ = j-1
  M.rlim_ = j+1
end

"""
inner(y::MPS, A::MPO, x::MPS)

Compute <y|A|x>
"""
function inner(y::MPS,
               A::MPO,
               x::MPS)::Number
  N = length(A)
  if length(y) != N || length(x) != N
      throw(DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y))"))
  end
  ydag = dag(y)
  simlinks!(ydag)
  sAx = siteinds(A,x)
  replacesites!(ydag,sAx)
  O = ydag[1]*A[1]*x[1]
  @inbounds for j ∈ eachindex(y)[2:end]
    O = O*ydag[j]*A[j]*x[j]
  end
  return O[]
end

function plussers(left_ind::Index, right_ind::Index, sum_ind::Index)
    #if dir(left_ind) == dir(right_ind) == Neither
        total_dim    = dim(left_ind) + dim(right_ind)
        total_dim    = max(total_dim, 1)
        # TODO: I am not sure if we should be using delta
        # tensors for this purpose? I think we should consider
        # not allowing them to be made with different index sizes
        #left_tensor  = δ(left_ind, sum_ind)
        left_tensor  = diagITensor(1.0,left_ind, sum_ind)
        right_tensor = ITensor(right_ind, sum_ind)
        for i in 1:dim(right_ind)
            right_tensor[right_ind(i), sum_ind(dim(left_ind) + i)] = 1
        end
        return left_tensor, right_tensor
    #else # tensors have QNs
    #    throw(ArgumentError("support for adding MPOs with defined quantum numbers not implemented yet."))
    #end
end

function sum(A::T, B::T; kwargs...) where {T <: Union{MPS, MPO}}
    n = A.N_ 
    length(B) =! n && throw(DimensionMismatch("lengths of MPOs A ($n) and B ($(length(B))) do not match"))
    position!(A, 1, kwargs...)
    position!(B, 1, kwargs...)
    C = similar(A)
    rand_plev = 13124
    lAs = [linkindex(A, i) for i in 1:n-1]
    prime!(A, rand_plev, "Link")

    first  = fill(ITensor(), n)
    second = fill(ITensor(), n)
    for i in 1:n-1
        lA = linkindex(A, i)
        lB = linkindex(B, i)
        r = Index(dim(lA) + dim(lB), tags(lA))
        f, s = plussers(lA, lB, r)
        first[i] = f
        second[i] = s
    end
    C[1] = A[1] * first[1] + B[1] * second[1]
    for i in 2:n-1
        C[i] = dag(first[i-1]) * A[i] * first[i] + dag(second[i-1]) * B[i] * second[i]
    end
    C[n] = dag(first[n-1]) * A[n] + dag(second[n-1]) * B[n]
    prime!(C, -rand_plev, "Link")
    position!(C, 1, kwargs...)
    return C
end

function applyMPO(A::MPO, psi::MPS; kwargs...)::MPS
    method = get(kwargs, :method, "DensityMatrix")
    if method == "DensityMatrix"
        return densityMatrixApplyMPO(A, psi; kwargs...)
    end
    throw(ArgumentError("Method $method not supported"))
end

function densityMatrixApplyMPO(A::MPO, psi::MPS; kwargs...)::MPS
    n = length(A)
    n != length(psi) && throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(psi))) do not match"))
    psi_out = similar(psi)
    cutoff::Float64 = get(kwargs, :cutoff, 1e-13)
    maxdim::Int = get(kwargs,:maxdim,maxLinkDim(psi))
    mindim::Int = max(get(kwargs,:mindim,1), 1)
    normalize::Bool = get(kwargs, :normalize, false) 

    all(x->x!=Index(), [siteindex(A, psi, j) for j in 1:n]) || throw(ErrorException("MPS and MPO have different site indices in applyMPO method 'DensityMatrix'"))
    rand_plev = 14741

    psi_c = dag(copy(psi))
    A_c   = dag(copy(A))
    prime!(psi_c, rand_plev)
    prime!(A_c, rand_plev)

    for j in 1:n-1
        unique_site_ind = setdiff(findinds(A_c[j], "Site"), findindex(psi_c[j], "Site"))[1]
        A_c[j] = setprime(A_c[j], 1, unique_site_ind)
    end
    E = Vector{ITensor}(undef, n-1)
    E[1] = psi[1]*A[1]*A_c[1]*psi_c[1]
    for j in 2:n-1
        E[j] = E[j-1]*psi[j]*A[j]*A_c[j]*psi_c[j]
    end
    O     = psi[n] * A[n]
    ρ     = E[n-1] * O * dag(prime(O, rand_plev))
    ts    = tags(commonindex(psi[n], psi[n-1]))
    Lis   = commonindex(ρ, A[n])
    Ris   = uniqueinds(ρ, Lis)
    FU, D = eigen(ρ, Lis, Ris; tags=ts, kwargs...)
    psi_out[n] = setprime(dag(FU), 0, "Site")
    O     = O * FU * psi[n-1] * A[n-1]
    O     = prime(O, -1, "Site")
    for j in reverse(2:n-1)
        dO  = prime(dag(O), rand_plev)
        ρ   = E[j-1] * O * dO
        ts  = tags(commonindex(psi[j], psi[j-1]))
        Lis = IndexSet(commonindex(ρ, A[j]), commonindex(ρ, psi_out[j+1])) 
        Ris = uniqueinds(ρ, Lis)
        FU, D = eigen(ρ, Lis, Ris; tags=ts, kwargs...)
        psi_out[j] = dag(FU)
        O = O * FU * psi[j-1] * A[j-1]
        O = prime(O, -1, "Site")
    end
    if normalize
        O /= norm(O)
    end
    psi_out[1] = O
    psi_out.llim_ = 0
    psi_out.rlim_ = 2
    return psi_out
end

function nmultMPO(A::MPO, B::MPO; kwargs...)::MPO
    cutoff::Float64 = get(kwargs, :cutoff, 1e-14)
    resp_degen::Bool = get(kwargs, :respect_degenerate, true) 
    maxdim::Int = get(kwargs,:maxdim,maxLinkDim(A)*maxLinkDim(B))
    mindim::Int = max(get(kwargs,:mindim,1), 1)
    N = length(A)
    N != length(B) && throw(DimensionMismatch("lengths of MPOs A ($N) and B ($(length(B))) do not match"))
    A_ = copy(A)
    position!(A_, 1)
    B_ = copy(B)
    position!(B_, 1)

    links_A = findinds.(A.A_, "Link")
    links_B = findinds.(B.A_, "Link")

    for i in 1:N
        if length(commoninds(findinds(A_[i], "Site"), findinds(B_[i], "Site"))) == 2
            A_[i] = prime(A_[i], "Site")
        end
    end
    res = deepcopy(A_)
    for i in 1:N-1
        ci = commonindex(res[i], res[i+1])
        new_ci = Index(dim(ci), tags(ci))
        replaceindex!(res[i], ci, new_ci)
        replaceindex!(res[i+1], ci, new_ci)
        @assert commonindex(res[i], res[i+1]) != commonindex(A[i], A[i+1])
    end
    sites_A = [setdiff(findinds(x, "Site"), findindex(y, "Site"))[1] for (x,y) in zip(tensors(A_), tensors(B_))]
    sites_B = [setdiff(findinds(x, "Site"), findindex(y, "Site"))[1] for (x,y) in zip(tensors(B_), tensors(A_))]
    res[1] = ITensor(sites_A[1], sites_B[1], commonindex(res[1], res[2]))
    for i in 1:N-2
        if i == 1
            clust = A_[i] * B_[i]
        else
            clust = nfork * A_[i] * B_[i]
        end
        lA = commonindex(A_[i], A_[i+1])
        lB = commonindex(B_[i], B_[i+1])
        nfork = ITensor(lA, lB, commonindex(res[i], res[i+1]))
        res[i], nfork = factorize(mapprime(clust,2,1), inds(res[i]), dir="fromleft", tags=tags(lA), cutoff=cutoff, maxdim=maxdim, mindim=mindim)
        mid = dag(commonindex(res[i], nfork))
        res[i+1] = ITensor(mid, sites_A[i+1], sites_B[i+1], commonindex(res[i+1], res[i+2]))
    end
    clust = nfork * A_[N-1] * B_[N-1]
    nfork = clust * A_[N] * B_[N]

    # in case we primed A
    A_ind = uniqueindex(findinds(A_[N-1], "Site"), findinds(B_[N-1], "Site"))
    Lis = IndexSet(A_ind, sites_B[N-1], commonindex(res[N-2], res[N-1]))
    U, V, ci = factorize(nfork,Lis,dir="fromright",cutoff=cutoff,which_factorization="svd",tags="Link,n=$(N-1)",maxdim=maxdim,mindim=mindim)
    res[N-1] = U
    res[N] = V
    position!(res, 1)
    for i in 1:N
        res[i] = mapprime(res[i], 2, 1)
    end
    return res
end
