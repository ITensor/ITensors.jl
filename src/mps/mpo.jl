export MPO,
       randomMPO

struct MPO{T}
  N_::Int
  A_::Vector{ITensor{T}}

  MPO() = new{Dense{Nothing}}(0,Vector{ITensor{Dense{Nothing}}}())

  function MPO(N::Int, A::Vector{ITensor{T}})
    new{T}(N,A)
  end

  function MPO(sites)
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
    new{store(v[1])}(N,v)
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
    new{store(its)}(N,its)
  end

  function MPO(sites::SiteSet, 
               ops::String)
      MPO(sites, fill(ops, length(sites)))
  end

end

function randomMPO(sites,
                   m::Int=1)
  M = MPO(sites)
  @inbounds for i ∈ eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  if m > 1
    error("randomMPS: currently only m==1 supported")
  end
  return M
end

length(m::MPO) = m.N_

getindex(m::MPO, n::Integer) = getindex(m.A_,n)
setindex!(m::MPO,T::ITensor,n::Integer) = setindex!(m.A_,T,n)

copy(m::MPO) = MPO(m.N_,copy(m.A_))

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

function maxDim(M::T) where {T <: Union{MPS,MPO}}
  md = 0
  for b ∈ eachindex(M)[1:end-1] 
    md = max(md,dim(linkindex(M,b)))
  end
  md
end

function show(io::IO, W::MPO)
  print(io,"MPO")
  (length(W) > 0) && print(io,"\n")
  @inbounds for (i, w) ∈ enumerate(inds.(W.A_))
    println(io,"$i  $w")
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

