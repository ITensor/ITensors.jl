export MPO,
       randomMPO,
       applyMPO,
       multMPO,
       errorMPOprod,
       maxlinkdim,
       orthogonalize!,
       truncate!,
       sum

mutable struct MPO
  N_::Int
  A_::Vector{ITensor}
  llim_::Int
  rlim_::Int

  MPO() = new(0,Vector{ITensor}(), 0, 0)

  function MPO(N::Int, A::Vector{<:ITensor}, llim::Int=0, rlim::Int=N+1)
    new(N, A, llim, rlim)
  end

  function MPO(sites::Vector{Index})
    N = length(sites)
    v = Vector{ITensor}(undef, N)
    l = [Index(1, "Link,l=$ii") for ii ∈ 1:N-1]
    for ii ∈ eachindex(sites)
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
 
end

MPO(N::Int) = MPO(N,fill(ITensor(),N))

function MPO(sites,
             ops::Vector{String})
  N = length(sites)
  its = Vector{ITensor}(undef, N)
  links = Vector{Index}(undef, N)
  for ii ∈ eachindex(sites)
    si = sites[ii]
    d = dim(si)
    spin_op = op(sites, ops[ii], ii)
    links[ii] = Index(1, "Link,n=$ii")
    local this_it
    if ii == 1
      this_it = ITensor(links[ii], si, si')
      for jj in 1:d, jjp in 1:d
        this_it[links[ii](1), si[jj], si'[jjp]] = spin_op[si[jj], si'[jjp]]
      end
    elseif ii == N
      this_it = ITensor(links[ii-1], si, si')
      for jj in 1:d, jjp in 1:d
        this_it[links[ii-1](1), si[jj], si'[jjp]] = spin_op[si[jj], si'[jjp]]
      end
    else
      this_it = ITensor(links[ii-1], links[ii], si, si')
      for jj in 1:d, jjp in 1:d
        this_it[links[ii-1](1), links[ii](1), si[jj], si'[jjp]] = spin_op[si[jj], si'[jjp]]
      end
    end
    its[ii] = this_it
  end
  MPO(N,its)
end

MPO(sites, ops::String) = MPO(sites, fill(ops, length(sites)))

function randomMPO(sites, m::Int=1)
  M = MPO(sites)
  for i ∈ eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  m > 1 && throw(ArgumentError("randomMPO: currently only m==1 supported"))
  return M
end

Base.length(m::MPO) = m.N_
tensors(m::MPO) = m.A_
leftlim(m::MPO) = m.llim_
rightlim(m::MPO) = m.rlim_

function set_leftlim!(m::MPO,new_ll::Int)
  m.llim_ = new_ll
end

function set_rightlim!(m::MPO,new_rl::Int)
  m.rlim_ = new_rl
end

Base.getindex(m::MPO, n::Integer) = getindex(tensors(m), n)

function Base.setindex!(M::MPO,T::ITensor,n::Integer)
  (n <= leftlim(M)) && set_leftlim!(M,n-1)
  (n >= rightlim(M)) && set_rightlim!(M,n+1)
  setindex!(tensors(M),T,n)
end

Base.copy(m::MPO) = MPO(m.N_, copy(tensors(m)))
Base.similar(m::MPO) = MPO(m.N_, similar(tensors(m)), 0, m.N_)

function Base.deepcopy(m::T) where {T <: Union{MPO,MPS}}
    res = similar(m)
    # otherwise we will end up modifying the elements of A!
    res.A_ = deepcopy(tensors(m))
    return res
end

Base.eachindex(m::MPO) = 1:length(m)

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

siteinds(A::MPO,x::MPS) = [siteindex(A,x,j) for j ∈ 1:length(A)]

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
maxlinkdim(M::MPS)
maxlinkdim(M::MPO)

Get the maximum link dimension of the MPS or MPO.
"""
function maxlinkdim(M::T) where {T <: Union{MPS,MPO}}
  md = 0
  for b ∈ eachindex(M)[1:end-1]
    md = max(md,dim(linkindex(M,b)))
  end
  md
end

function Base.show(io::IO, W::MPO)
  print(io,"MPO")
  (length(W) > 0) && print(io,"\n")
  for (i, A) ∈ enumerate(tensors(W))
    if order(A) != 0
      println(io,"[$i] $(inds(A))")
    else
      println(io,"[$i] ITensor()")
    end
  end
end

function linkindex(M::MPO,j::Integer)
  N = length(M)
  j ≥ length(M) && error("No link index to the right of site $j (length of MPO is $N)")
  li = commonindex(M[j],M[j+1])
  if isnothing(li)
    error("linkindex: no MPO link index at link $j")
  end
  return li
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

"""
inner(B::MPO, y::MPS, A::MPO, x::MPS)

Compute <By|A|x>
"""
function inner(B::MPO,
               y::MPS,
               A::MPO,
               x::MPS)::Number
  N = length(B)
  if length(y) != N || length(x) != N || length(A) != N
      throw(DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y)) or $(length(A))"))
  end
  ydag = dag(y)
  prime!(ydag, 2)
  Bdag = dag(B)
  prime!(Bdag)
  # Swap prime levels 1 -> 2 and 2 -> 1.
  @inbounds for j ∈ eachindex(Bdag)
    swapprime!(inds(Bdag[j]),2,3)
    swapprime!(inds(Bdag[j]),1,2)
    swapprime!(inds(Bdag[j]),3,1)
  end
  O = ydag[1]*Bdag[1]*A[1]*x[1]
  @inbounds for j ∈ eachindex(y)[2:end]
    O = O*ydag[j]*Bdag[j]*A[j]*x[j]
  end
  return O[]
end


"""
errorMPOprod(y::MPS, A::MPO, x::MPS)

Compute the distance between A|x> and an approximation MPS y:
| |y> - A|x> |/| A|x> | = √(1 + (<y|y> - 2*real(<y|A|x>))/<Ax|A|x>)
"""
function errorMPOprod(y::MPS, A::MPO, x::MPS)
  N = length(A)
  if length(y) != N || length(x) != N
      throw(DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y))"))
  end
  return sqrt(abs(1. + (inner(y,y) - 2*real(inner(y,A,x)))/inner(A,x,A,x)))
end

function plussers(::Type{T}, left_ind::Index, right_ind::Index, sum_ind::Index) where {T<:Array}
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

function Base.sum(A::T, B::T; kwargs...) where {T <: Union{MPS, MPO}}
    N = length(A)
    length(B) != N && throw(DimensionMismatch("lengths of MPOs A ($N) and B ($(length(B))) do not match"))
    orthogonalize!(A, 1; kwargs...)
    orthogonalize!(B, 1; kwargs...)
    C = similar(A)
    rand_plev = 13124
    lAs = [linkindex(A, i) for i in 1:N-1]
    prime!(A, rand_plev, "Link")

    first  = Vector{ITensor{2}}(undef,N-1)
    second = Vector{ITensor{2}}(undef,N-1)
    for i in 1:N-1
        lA = linkindex(A, i)
        lB = linkindex(B, i)
        r  = Index(dim(lA) + dim(lB), tags(lA))
        f, s = plussers(typeof(data(store(A[1]))), lA, lB, r)
        first[i]  = f
        second[i] = s
    end
    C[1] = A[1] * first[1] + B[1] * second[1]
    for i in 2:N-1
        C[i] = dag(first[i-1]) * A[i] * first[i] + dag(second[i-1]) * B[i] * second[i]
    end
    C[N] = dag(first[N-1]) * A[N] + dag(second[N-1]) * B[N]
    prime!(C, -rand_plev, "Link")
    truncate!(C; kwargs...)
    return C
end

function Base.sum(A::Vector{T}; kwargs...) where {T <: Union{MPS, MPO}}
    length(A) == 0 && return T()
    length(A) == 1 && return A[1]
    length(A) == 2 && return sum(A[1], A[2]; kwargs...)
    nsize = isodd(length(A)) ? (div(length(A) - 1, 2) + 1) : div(length(A), 2)
    newterms = Vector{T}(undef, nsize)
    np = 1
    for n in 1:2:length(A) - 1
      newterms[np] = sum(A[n], A[n+1]; kwargs...)
      np += 1
    end
    if isodd(length(A))
      newterms[nsize] = A[end]
    end
    return sum(newterms; kwargs...)
end

function applyMPO(A::MPO, psi::MPS; kwargs...)::MPS
  method = get(kwargs, :method, "DensityMatrix")
  if method == "DensityMatrix" || method == "densitymatrix"
    return densityMatrixApplyMPO(A, psi; kwargs...)
  elseif method == "naive" || method == "Naive"
    return naiveApplyMPO(A, psi; kwargs...)
  end
  throw(ArgumentError("Method $method not supported"))
end

function densityMatrixApplyMPO(A::MPO, psi::MPS; kwargs...)::MPS
    n = length(A)
    n != length(psi) && throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(psi))) do not match"))
    psi_out         = similar(psi)
    cutoff::Float64 = get(kwargs, :cutoff, 1e-13)

    maxdim::Int     = get(kwargs,:maxdim,maxlinkdim(psi))
    mindim::Int     = max(get(kwargs,:mindim,1), 1)
    normalize::Bool = get(kwargs, :normalize, false) 
    all(x -> x != Index(), [siteindex(A, psi, j) for j in 1:n]) || throw(ErrorException("MPS and MPO have different site indices in applyMPO method 'DensityMatrix'"))

    rand_plev = 14741
    psi_c     = dag(copy(psi))
    A_c       = dag(copy(A))
    prime!(psi_c, rand_plev)
    prime!(A_c, rand_plev)
    for j in 1:n-1
        unique_site_ind = setdiff(findinds(A_c[j], "Site"), findindex(psi_c[j], "Site"))[1]
        pl = id(unique_site_ind) == id(commonindex(A_c[j], psi_c[j])) ? 1 : 0
        A_c[j] = setprime(A_c[j], pl, unique_site_ind)
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
    FU, D = eigenHermitian(ρ, Lis, Ris; ispossemidef=true, 
                                        tags=ts, 
                                        kwargs...)
    psi_out[n] = setprime(dag(FU), 0, "Site")
    O     = O * FU * psi[n-1] * A[n-1]
    O     = prime(O, -1, "Site")
    for j in reverse(2:n-1)
        dO  = prime(dag(O), rand_plev)
        ρ   = E[j-1] * O * dO
        ts  = tags(commonindex(psi[j], psi[j-1]))
        Lis = IndexSet(commonindex(ρ, A[j]), commonindex(ρ, psi_out[j+1])) 
        Ris = uniqueinds(ρ, Lis)
        FU, D = eigenHermitian(ρ, Lis, Ris; ispossemidef=true,
                                            tags=ts, 
                                            kwargs...)
        psi_out[j] = dag(FU)
        O = O * FU * psi[j-1] * A[j-1]
        O = prime(O, -1, "Site")
    end
    if normalize
        O /= norm(O)
    end
    psi_out[1]    = copy(O)
    psi_out.llim_ = 0
    psi_out.rlim_ = 2
    return psi_out
end

function naiveApplyMPO(A::MPO, psi::MPS; kwargs...)::MPS
  N = length(A)
  if N != length(psi) 
    throw(DimensionMismatch("lengths of MPO ($N) and MPS ($(length(psi))) do not match"))
  end

  psi_out = MPS(N)
  for j=1:N
    psi_out[j] = noprime(A[j]*psi[j])
  end

  for b=1:(N-1)
    Al = commonindex(A[b],A[b+1])
    pl = commonindex(psi[b],psi[b+1])
    C,_ = combiner(Al,pl)
    psi_out[b] *= C
    psi_out[b+1] *= C
  end

  truncate!(psi_out;kwargs...)

  return psi_out
end

function multMPO(A::MPO, B::MPO; kwargs...)::MPO
    cutoff::Float64 = get(kwargs, :cutoff, 1e-14)
    resp_degen::Bool = get(kwargs, :respect_degenerate, true)
    maxdim::Int = get(kwargs,:maxdim,maxlinkdim(A)*maxlinkdim(B))
    mindim::Int = max(get(kwargs,:mindim,1), 1)
    N = length(A)
    N != length(B) && throw(DimensionMismatch("lengths of MPOs A ($N) and B ($(length(B))) do not match"))
    A_ = copy(A)
    orthogonalize!(A_, 1)
    B_ = copy(B)
    orthogonalize!(B_, 1)

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
    sites_A = Index[]
    sites_B = Index[]
    @inbounds for (AA, BB) in zip(tensors(A_), tensors(B_))
        sda = setdiff(findinds(AA, "Site"), findinds(BB, "Site"))
        sdb = setdiff(findinds(BB, "Site"), findinds(AA, "Site"))
        sda_ind = setprime(sda[1], 0) == sdb[1] ? plev(sda[1]) == 1 ? sda[1] : setprime(sda[1], 1) : setprime(sda[1], 0)
        push!(sites_A, sda_ind)
        push!(sites_B, sdb[1])
    end
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
    U, V = factorize(nfork,Lis,dir="fromright",cutoff=cutoff,which_factorization="svd",tags="Link,n=$(N-1)",maxdim=maxdim,mindim=mindim)
    res[N-1] = U
    res[N] = V
    truncate!(res;kwargs...)
    for i in 1:N
        res[i] = mapprime(res[i], 2, 1)
    end
    return res
end

function orthogonalize!(M::Union{MPS,MPO},
                        j::Int;
                        kwargs...)
  while leftlim(M) < (j-1)
    (leftlim(M) < 0) && set_leftlim!(M,0)
    b = leftlim(M)+1
    linds = uniqueinds(M[b],M[b+1])
    Q,R = qr(M[b], linds)
    M[b] = Q
    M[b+1] *= R
    set_leftlim!(M,b)
    if rightlim(M) < leftlim(M)+2
      set_rightlim!(M,leftlim(M)+2)
    end
  end

  N = length(M)

  while rightlim(M) > (j+1)
    (rightlim(M) > (N+1)) && set_rightlim!(M,N+1)
    b = rightlim(M)-2
    rinds = uniqueinds(M[b+1],M[b])
    Q,R = qr(M[b+1], rinds)
    M[b+1] = Q
    M[b] *= R
    set_rightlim!(M,b+1)
    if leftlim(M) > rightlim(M)-2
      set_leftlim!(M,rightlim(M)-2)
    end
  end
end

function Tensors.truncate!(M::Union{MPS,MPO}; kwargs...)
  N = length(M)

  # Left-orthogonalize all tensors to make
  # truncations controlled
  orthogonalize!(M,N)

  # Perform truncations in a right-to-left sweep
  for j in reverse(2:N)
    rinds = uniqueinds(M[j],M[j-1])
    U,S,V = svd(M[j],rinds;kwargs...)
    M[j] = U
    M[j-1] *= (S*V)
    set_rightlim!(M,j)
  end

end

@doc """
orthogonalize!(M::MPS, j::Int; kwargs...)

Move the orthogonality center of the MPS
to site j. No observable property of the
MPS will be changed, and no truncation of the
bond indices is performed. Afterward, tensors
1,2,...,j-1 will be left-orthogonal and tensors
j+1,j+2,...,N will be right-orthogonal.

orthogonalize!(W::MPO, j::Int; kwargs...)

Move the orthogonality center of an MPO to site j.
""" orthogonalize!

@doc """
truncate!(M::MPS; kwargs...)

Perform a truncation of all bonds of an MPS,
using the truncation parameters (cutoff,maxdim, etc.)
provided as keyword arguments.

truncate!(M::MPO; kwargs...)

Perform a truncation of all bonds of an MPO,
using the truncation parameters (cutoff,maxdim, etc.)
provided as keyword arguments.
""" truncate!
