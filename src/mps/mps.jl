export MPS,
       position!,
       prime!,
       primelinks!,
       simlinks!,
       inner,
       randomMPS,
       maxDim,
       linkindex,
       siteindex,
       siteinds

mutable struct MPS{T <: TensorStorage}
    N_::Int
    A_::Vector{ITensor{T}}
    llim_::Int
    rlim_::Int
    MPS(N::Int,
        A::Vector{<:ITensor},
        llim::Int=0,
        rlim::Int=N+1) = new{TensorStorage}(N, A, llim, rlim)
    MPS{T}(N::Int,
           A::Vector{ITensor{T}},
           llim::Int=0,
           rlim::Int=N+1) where {T <: TensorStorage} = new{T}(N, A, llim, rlim)
end

MPS() = MPS(0,Vector{ITensor{<:TensorStorage}}(),0,0)

function MPS(sites::SiteSet)
    N = length(sites)
    v = Vector{ITensor{<:TensorStorage}}(undef, N)
    l = [Index(1, "Link,l=$ii") for ii=1:N-1]
    @inbounds for ii in eachindex(sites)
        s = sites[ii]
        if ii == 1
            v[ii] = ITensor(l[ii], s)
        elseif ii == N
            v[ii] = ITensor(l[ii-1], s)
        else
            v[ii] = ITensor(l[ii-1], l[ii], s)
        end
    end
    MPS(N,v)
end

function MPS(is::InitState)
    N = length(is)
    As = Vector{ITensor{<:TensorStorage}}(undef,N)
    links  = Vector{Index}(undef,N)
    @inbounds for n in eachindex(is)
        s = sites(is)[n]
        links[n] = Index(1,"Link,l=$n")
        if n == 1
            A = ITensor(s,links[n])
            A[state(is,n),links[n](1)] = 1.0
        elseif n == N
            A = ITensor(links[n-1],s)
            A[links[n-1](1),state(is,n)] = 1.0
        else
            A = ITensor(links[n-1],links[n],s)
            A[links[n-1](1),state(is,n),links[n](1)] = 1.0
        end
        As[n] = A
    end
    MPS(N,As,0,2)
end

MPS(N::Int, d::Int, opcode::String) = MPS(InitState(Sites(N,d), opcode))
MPS(N::Int) = MPS(N,Vector{ITensor{TensorStorage}}(undef, N),0,N+1)
MPS(s::SiteSet, opcode::String) = MPS(InitState(s, opcode))

function randomMPS(sites)
  M = MPS(sites)
  @inbounds for i ∈ eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  M.llim_ = 1
  M.rlim_ = length(M)
  return M
end

length(m::MPS) = m.N_
leftLim(m::MPS) = m.llim_
rightLim(m::MPS) = m.rlim_

getindex(m::MPS, n::Integer) = getindex(m.A_,n)
setindex!(m::MPS,T::ITensor,n::Integer) = setindex!(m.A_,T,n)

copy(m::MPS) = MPS(m.N_,copy(m.A_),m.llim_,m.rlim_)

eachindex(m::MPS) = 1:length(m)


"""
    dag(m::MPS)

Hermitian conjugation of a matrix product state `m`.
"""
function dag(m::MPS)
  N = length(m)
  mdag = MPS(N)
  @inbounds for i ∈ eachindex(m)
    mdag[i] = dag(m[i])
  end
  return mdag
end

function show(io::IO, M::MPS)
  print(io,"MPS")
  (length(M) > 0) && print(io,"\n")
  @inbounds for (i, m) ∈ enumerate(inds.(M.A_))
    println(io,"$i  $m")
  end
end

function linkindex(M::MPS,j::Integer) 
  N = length(M)
  j ≥ length(M) && error("No link index to the right of site $j (length of MPS is $N)")
  li = commonindex(M[j],M[j+1])
  if isdefault(li)
    error("linkindex: no MPS link index at link $j")
  end
  return li
end

function siteindex(M::MPS,j::Integer)
  N = length(M)
  if j == 1
    si = uniqueindex(M[j],M[j+1])
  elseif j == N
    si = uniqueindex(M[j],M[j-1])
  else
    si = uniqueindex(M[j],(M[j-1],M[j+1]))
  end
  return si
end

function siteinds(M::MPS)
  N = length(M)
  is = IndexSet(N)
  @inbounds for j ∈ eachindex(M)
    is[j] = siteindex(M,j)
  end
  return is
end

function replacesites!(M::MPS,sites)
  N = length(M)
  @inbounds for j in eachindex(M)
    sj = siteindex(M,j)
    replaceindex!(M[j],sj,sites[j])
  end
  return M
end

function position!(M::MPS,
                   j::Integer)
  N = length(M)
  while leftLim(M) < (j-1)
    ll = leftLim(M)+1
    s = findindex(M[ll],"Site")
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
    s = findindex(M[rl],"Site")
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
inner(ψ::MPS, ϕ::MPS)

Compute <ψ|ϕ>
"""
function inner(M1::MPS, M2::MPS)::Number
  N = length(M1)
  if length(M2) != N
      throw(DimensionMismatch("inner: mismatched lengths $N and $(length(M2))"))
  end
  M1dag = dag(M1)
  simlinks!(M1dag)
  O = M1dag[1]*M2[1]
  @inbounds for j ∈ eachindex(M1)[2:end]
    O *= M1dag[j]*M2[j]
  end
  return O[]
end

function replaceBond!(M::MPS,
                      b::Int,
                      phi::ITensor;
                      kwargs...)
  FU,FV = factorize(phi,inds(M[b]); which_factorization="automatic", kwargs...)
  M[b]   = FU
  M[b+1] = FV
end

