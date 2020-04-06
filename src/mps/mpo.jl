export MPO,
       randomMPO,
       applympo,
       multmpo,
       error_mpoprod,
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

  function MPO(sites::Vector{<:Index})
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

MPO(A::Vector{<:ITensor}) = MPO(length(A),A,0,length(A)+1)

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
Tensors.store(m::MPO) = m.A_
leftlim(m::MPO) = m.llim_
rightlim(m::MPO) = m.rlim_

function setleftlim!(m::MPO,new_ll::Int)
  m.llim_ = new_ll
end

function setrightlim!(m::MPO,new_rl::Int)
  m.rlim_ = new_rl
end

Base.getindex(m::MPO, n::Integer) = getindex(store(m), n)

function Base.setindex!(M::MPO,T::ITensor,n::Integer)
  (n <= leftlim(M)) && setleftlim!(M,n-1)
  (n >= rightlim(M)) && setrightlim!(M,n+1)
  setindex!(store(M),T,n)
end

Base.copy(m::MPO) = MPO(m.N_, copy(store(m)))
Base.similar(m::MPO) = MPO(m.N_, similar(store(m)), 0, m.N_)

function Base.deepcopy(m::T) where {T <: Union{MPO,MPS}}
    res = similar(m)
    # otherwise we will end up modifying the elements of A!
    res.A_ = deepcopy(store(m))
    return res
end

const MPSorMPO = Union{MPS,MPO}

Base.eachindex(m::MPSorMPO) = 1:length(m)
Base.iterate(M::MPSorMPO) = iterate(store(M))
Base.iterate(M::MPSorMPO, state) = iterate(store(M), state)

# TODO: optimize finding the index a little bit
# First do: scom = commonind(A[j],x[j])
# Then do: uniqueind(A[j],A[j-1],A[j+1],(scom,))
function siteindex(A::MPO,x::MPS,j::Integer)
  N = length(A)
  if j == 1
    si = uniqueind(A[j],A[j+1],x[j])
  elseif j == N
    si = uniqueind(A[j],A[j-1],x[j])
  else
    si = uniqueind(A[j],A[j-1],A[j+1],x[j])
  end
  return si
end

siteinds(A::MPO,x::MPS) = [siteindex(A,x,j) for j ∈ 1:length(A)]

"""
    dag(m::MPS)
    dag(m::MPO)

Hermitian conjugation of a matrix product state or operator `m`.
"""

function Tensors.dag(m::T) where {T <: Union{MPS, MPO}}
  N = length(m)
  mdag = T(N)
  for i ∈ eachindex(m)
    mdag[i] = dag(m[i])
  end
  return mdag
end

function prime!(M::Union{MPS,MPO},vargs...)
  for i ∈ eachindex(M)
    prime!(M[i],vargs...)
  end
end

function primelinks!(M::Union{MPS,MPO}, plinc::Integer = 1)
  for i ∈ eachindex(M)[1:end-1]
    l = linkind(M,i)
    prime!(M[i],plinc,l)
    prime!(M[i+1],plinc,l)
  end
end

function simlinks!(M::Union{MPS,MPO})
  for i ∈ eachindex(M)[1:end-1]
    isnothing(commonind(M[i],M[i+1])) && continue
    l = linkind(M,i)
    l̃ = sim(l)
    replaceind!(M[i],l,l̃)
    replaceind!(M[i+1],l,dag(l̃))
  end
end

"""
maxlinkdim(M::MPS)
maxlinkdim(M::MPO)

Get the maximum link dimension of the MPS or MPO.
"""
function maxlinkdim(M::Union{MPS,MPO})
  md = 0
  for b ∈ eachindex(M)[1:end-1]
    md = max(md,dim(linkind(M,b)))
  end
  md
end

function Base.show(io::IO, W::MPO)
  print(io,"MPO")
  (length(W) > 0) && print(io,"\n")
  for (i, A) ∈ enumerate(store(W))
    if order(A) != 0
      println(io,"[$i] $(inds(A))")
    else
      println(io,"[$i] ITensor()")
    end
  end
end

function linkind(M::MPO,j::Integer)
  N = length(M)
  j ≥ length(M) && error("No link index to the right of site $j (length of MPO is $N)")
  li = commonind(M[j],M[j+1])
  if isnothing(li)
    error("linkind: no MPO link index at link $j")
  end
  return li
end


"""
dot(y::MPS, A::MPO, x::MPS)
inner(y::MPS, A::MPO, x::MPS)

Compute <y|A|x>
"""
function LinearAlgebra.dot(y::MPS,
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
dot(B::MPO, y::MPS, A::MPO, x::MPS)
inner(B::MPO, y::MPS, A::MPO, x::MPS)

Compute <By|A|x>
"""
function LinearAlgebra.dot(B::MPO,
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
  for j ∈ eachindex(Bdag)
    Axcommon = commonind(A[j], x[j])
    ABcommon = uniqueind(inds(A[j], "Site"), IndexSet(Axcommon))
    swapprime!(Bdag[j],2,3)
    swapprime!(Bdag[j],1,2)
    swapprime!(Bdag[j],3,1)
    noprime!(Bdag[j],prime(ABcommon,2))
  end
  yB = ydag[1] * Bdag[1]
  Ax = A[1] * x[1]
  O = yB*Ax
  for j ∈ eachindex(y)[2:end]
    yB = ydag[j] * Bdag[j]
    Ax = A[j] * x[j]
    yB *= O
    O = yB * Ax 
  end
  return O[]
end


"""
error_mpoprod(y::MPS, A::MPO, x::MPS)

Compute the distance between A|x> and an approximation MPS y:
| |y> - A|x> |/| A|x> | = √(1 + (<y|y> - 2*real(<y|A|x>))/<Ax|A|x>)
"""
function error_mpoprod(y::MPS, A::MPO, x::MPS)
  N = length(A)
  if length(y) != N || length(x) != N
      throw(DimensionMismatch("inner: mismatched lengths $N and $(length(x)) or $(length(y))"))
  end
  iyy = inner(y,y)
  iyax = inner(y,A,x)
  iaxax = inner(A, x, A, x) 
  return sqrt(abs(1. + (iyy - 2*real(iyax))/iaxax))
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
    lAs = [linkind(A, i) for i in 1:N-1]
    prime!(A, rand_plev, "Link")

    first  = Vector{ITensor{2}}(undef,N-1)
    second = Vector{ITensor{2}}(undef,N-1)
    for i in 1:N-1
        lA = linkind(A, i)
        lB = linkind(B, i)
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

function applympo(A::MPO, psi::MPS; kwargs...)::MPS
  method = get(kwargs, :method, "densitymatrix")
  if method == "DensityMatrix"
    @warn "In applympo, method DensityMatrix is deprecated in favor of densitymatrix"
    method = "densitymatrix"
  end
  if method == "densitymatrix"
    return applympo_densitymatrix(A, psi; kwargs...)
  elseif method == "naive" || method == "Naive"
    return applympo_naive(A, psi; kwargs...)
  end
  throw(ArgumentError("Method $method not supported"))
end

function applympo_densitymatrix(A::MPO, psi::MPS; kwargs...)::MPS
  n = length(A)
  n != length(psi) && throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(psi))) do not match"))
  psi_out         = similar(psi)
  cutoff::Float64 = get(kwargs, :cutoff, 1e-13)

  maxdim::Int     = get(kwargs,:maxdim,maxlinkdim(psi))
  mindim::Int     = max(get(kwargs,:mindim,1), 1)
  normalize::Bool = get(kwargs, :normalize, false) 
  all(x -> x != Index(), [siteindex(A, psi, j) for j in 1:n]) || throw(ErrorException("MPS and MPO have different site indices in applympo method 'DensityMatrix'"))

  rand_plev = 14741
  psi_c     = dag(copy(psi))
  A_c       = dag(copy(A))
  prime!(psi_c, rand_plev)
  prime!(A_c, rand_plev)
  for j in 1:n
    s = siteindex(A,psi,j)
    s_dag = siteindex(A_c,psi_c,j)
    replaceind!(A_c[j],s_dag,s)
  end
  E = Vector{ITensor}(undef, n-1)
  E[1] = psi[1]*A[1]*A_c[1]*psi_c[1]
  for j in 2:n-1
    E[j] = E[j-1]*psi[j]*A[j]*A_c[j]*psi_c[j]
  end
  O     = psi[n] * A[n]
  ρ     = E[n-1] * O * dag(prime(O, rand_plev))
  ts    = tags(commonind(psi[n], psi[n-1]))
  Lis   = commonind(ρ, A[n])
  Ris   = prime(Lis, rand_plev)
  FU, D = eigen(ρ, Lis, Ris; ishermitian=true, 
                             tags=ts, 
                             kwargs...)
  psi_out[n] = setprime(dag(FU), 0, "Site")
  O     = O * FU * psi[n-1] * A[n-1]
  O     = noprime(O, "Site")
  for j in reverse(2:n-1)
    dO  = prime(dag(O), rand_plev)
    ρ   = E[j-1] * O * dO
    ts  = tags(commonind(psi[j], psi[j-1]))
    Lis = IndexSet(commonind(ρ, A[j]), commonind(ρ, psi_out[j+1])) 
    Ris = prime(Lis, rand_plev)
    FU, D = eigen(ρ, Lis, Ris; ishermitian=true,
                               tags=ts, 
                               kwargs...)
    psi_out[j] = dag(FU)
    O = O * FU * psi[j-1] * A[j-1]
    O = noprime(O, "Site")
  end
  if normalize
    O /= norm(O)
  end
  psi_out[1]    = copy(O)
  psi_out.llim_ = 0
  psi_out.rlim_ = 2
  return psi_out
end

function applympo_naive(A::MPO, psi::MPS; kwargs...)::MPS
  truncate = get(kwargs,:truncate,true)

  N = length(A)
  if N != length(psi) 
    throw(DimensionMismatch("lengths of MPO ($N) and MPS ($(length(psi))) do not match"))
  end

  psi_out = MPS(N)
  for j=1:N
    psi_out[j] = noprime(A[j]*psi[j])
  end

  for b=1:(N-1)
    Al = commonind(A[b],A[b+1])
    pl = commonind(psi[b],psi[b+1])
    C,_ = combiner(Al,pl)
    psi_out[b] *= C
    psi_out[b+1] *= dag(C)
  end

  if truncate
    truncate!(psi_out;kwargs...)
  end

  return psi_out
end

function multmpo(A::MPO, B::MPO; kwargs...)::MPO
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

    links_A = inds.(A.A_, "Link")
    links_B = inds.(B.A_, "Link")

    for i in 1:N
        if length(intersect(inds(A_[i], "Site"), inds(B_[i], "Site"))) == 2
            A_[i] = prime(A_[i], "Site")
        end
    end
    res = deepcopy(A_)
    for i in 1:N-1
        ci = commonind(res[i], res[i+1])
        new_ci = Index(dim(ci), tags(ci))
        replaceind!(res[i], ci, new_ci)
        replaceind!(res[i+1], ci, new_ci)
        @assert commonind(res[i], res[i+1]) != commonind(A[i], A[i+1])
    end
    sites_A = Index[]
    sites_B = Index[]
    @inbounds for (AA, BB) in zip(store(A_), store(B_))
        sda = setdiff(inds(AA, "Site"), inds(BB, "Site"))
        sdb = setdiff(inds(BB, "Site"), inds(AA, "Site"))
        sda_ind = setprime(sda[1], 0) == sdb[1] ? plev(sda[1]) == 1 ? sda[1] : setprime(sda[1], 1) : setprime(sda[1], 0)
        push!(sites_A, sda_ind)
        push!(sites_B, sdb[1])
    end
    res[1] = ITensor(sites_A[1], sites_B[1], commonind(res[1], res[2]))
    for i in 1:N-2
        if i == 1
            clust = A_[i] * B_[i]
        else
            clust = nfork * A_[i] * B_[i]
        end
        lA = commonind(A_[i], A_[i+1])
        lB = commonind(B_[i], B_[i+1])
        nfork = ITensor(lA, lB, commonind(res[i], res[i+1]))
        res[i], nfork = factorize(mapprime(clust,2,1),
                                  inds(res[i]),
                                  ortho="left",
                                  tags=tags(lA),
                                  cutoff=cutoff,
                                  maxdim=maxdim,
                                  mindim=mindim)
        mid = dag(commonind(res[i], nfork))
        res[i+1] = ITensor(mid,
                           sites_A[i+1],
                           sites_B[i+1],
                           commonind(res[i+1], res[i+2]))
    end
    clust = nfork * A_[N-1] * B_[N-1]
    nfork = clust * A_[N] * B_[N]

    # in case we primed A
    A_ind = uniqueind(inds(A_[N-1], "Site"), inds(B_[N-1], "Site"))
    Lis = IndexSet(A_ind, sites_B[N-1], commonind(res[N-2], res[N-1]))
    U, V = factorize(nfork, Lis; 
                     ortho="right",
                     cutoff=cutoff,
                     tags="Link,n=$(N-1)",
                     maxdim=maxdim,
                     mindim=mindim)
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
    (leftlim(M) < 0) && setleftlim!(M,0)
    b = leftlim(M)+1
    linds = uniqueinds(M[b],M[b+1])
    L,R = factorize(M[b], linds)
    M[b] = L
    M[b+1] *= R

    setleftlim!(M,b)
    if rightlim(M) < leftlim(M)+2
      setrightlim!(M,leftlim(M)+2)
    end
  end

  N = length(M)

  while rightlim(M) > (j+1)
    (rightlim(M) > (N+1)) && setrightlim!(M,N+1)
    b = rightlim(M)-2
    rinds = uniqueinds(M[b+1],M[b])
    L,R = factorize(M[b+1], rinds)
    M[b+1] = L
    M[b] *= R

    setrightlim!(M,b+1)
    if leftlim(M) > rightlim(M)-2
      setleftlim!(M,rightlim(M)-2)
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
    setrightlim!(M,j)
  end

end

# TODO: scale the tensors between the left limit
# and right limit by x^(1/N)
# where N is the distance between the left limit
# and right limit
function Base.:*(x::Number,M::Union{MPS,MPO})
  N = deepcopy(M)
  c = div(length(N), 2)
  N[c] .*= x
  return N
end

Base.:-(M::Union{MPS,MPO}) = Base.:*(-1,M)

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

@deprecate applyMPO(args...; kwargs...) applympo(args...; kwargs...)
@deprecate errorMPOprod(args...; kwargs...) error_mpoprod(args...; kwargs...)
@deprecate densityMatrixApplyMPO(args...; kwargs...) applympo_densitymatrix(args...; kwargs...)
@deprecate naiveApplyMPO(args...; kwargs...) applympo_naive(args...; kwargs...)
@deprecate multMPO(args...; kwargs...) multmpo(args...; kwargs...)
@deprecate set_leftlim!(args...; kwargs...) setleftlim!(args...; kwargs...)
@deprecate set_rightlim!(args...; kwargs...) setrightlim!(args...; kwargs...)
@deprecate tensors(args...; kwargs...) store(args...; kwargs...)

