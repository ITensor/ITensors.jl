export MPS,
       sample,
       sample!,
       leftlim,
       prime!,
       primelinks!,
       simlinks!,
       inner,
       isortho,
       productMPS,
       randomMPS,
       replacebond!,
       rightlim,
       linkindex,
       siteindex,
       siteinds


mutable struct MPS
  N_::Int
  A_::Vector{ITensor}
  llim_::Int
  rlim_::Int

  MPS() = new(0,Vector{ITensor}(),0,0)

  MPS(N::Int) = MPS(N,fill(ITensor(),N),0,N+1)

  function MPS(N::Int, 
               A::Vector{<:ITensor}, 
               llim::Int=0, 
               rlim::Int=N+1)
    new(N,A,llim,rlim)
  end
end

function MPS(sites)
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  l = [Index(1, "Link,l=$ii") for ii=1:N-1]
  for ii in eachindex(sites)
    s = sites[ii]
    if ii == 1
      v[ii] = ITensor(l[ii], s)
    elseif ii == N
      v[ii] = ITensor(l[ii-1], s)
    else
      v[ii] = ITensor(l[ii-1],s,l[ii])
    end
  end
  return MPS(N,v,0,N+1)
end

Base.length(m::MPS) = m.N_

# TODO: make this vec?
tensors(m::MPS) = m.A_

leftlim(m::MPS) = m.llim_
rightlim(m::MPS) = m.rlim_

function set_leftlim!(m::MPS,new_ll::Int) 
  m.llim_ = new_ll
end

function set_rightlim!(m::MPS,new_rl::Int) 
  m.rlim_ = new_rl
end

isortho(m::MPS) = (leftlim(m)+1 == rightlim(m)-1)

function orthoCenter(m::MPS)
  !isortho(m) && error("MPS has no well-defined orthogonality center")
  return leftlim(m)+1
end

Base.getindex(M::MPS, n::Integer) = getindex(tensors(M),n)

function Base.setindex!(M::MPS,T::ITensor,n::Integer) 
  (n <= leftlim(M)) && set_leftlim!(M,n-1)
  (n >= rightlim(M)) && set_rightlim!(M,n+1)
  setindex!(tensors(M),T,n)
end

Base.copy(m::MPS) = MPS(m.N_,copy(tensors(m)),m.llim_,m.rlim_)
Base.similar(m::MPS) = MPS(m.N_, similar(tensors(m)), 0, m.N_)

Base.eachindex(m::MPS) = 1:length(m)

function Base.show(io::IO, M::MPS)
  print(io,"MPS")
  (length(M) > 0) && print(io,"\n")
  for (i, A) ∈ enumerate(tensors(M))
    if order(A) != 0
      println(io,"[$i] $(inds(A))")
    else
      println(io,"[$i] ITensor()")
    end
  end
end

function randomMPS(sites)
  M = MPS(sites)
  for i in eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  M.llim_ = 1
  M.rlim_ = length(M)
  return M
end

function productMPS(ivals::Vector{IndexVal})
  N = length(ivals)
  As = Vector{ITensor}(undef,N)
  links  = Vector{Index}(undef,N)
  for n=1:N
    s = ind(ivals[n])
    links[n] = Index(1,"Link,l=$n")
    if n == 1
      A = ITensor(s,links[n])
      A[ivals[n],links[n](1)] = 1.0
    elseif n == N
      A = ITensor(links[n-1],s)
      A[links[n-1](1),ivals[n]] = 1.0
    else
      A = ITensor(links[n-1],s,links[n])
      A[links[n-1](1),ivals[n],links[n](1)] = 1.0
    end
    As[n] = A
  end
  return MPS(N,As,0,2)
end

function productMPS(sites,
                    states)
  if length(sites) != length(states)
    throw(DimensionMismatch("Number of sites and and initial states don't match"))
  end
  ivals = [state(sites[n],states[n]) for n=1:length(sites)]
  return productMPS(ivals)
end

function linkindex(M::MPS,j::Integer) 
  N = length(M)
  j ≥ length(M) && error("No link index to the right of site $j (length of MPS is $N)")
  li = commonindex(M[j],M[j+1])
  if isnothing(li)
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
  return [siteindex(M,j) for j in 1:length(M)]
end

function replacesites!(M::MPS,sites)
  for j in eachindex(M)
    sj = siteindex(M,j)
    replaceindex!(M[j],sj,sites[j])
  end
  return M
end

function inner(M1::MPS, M2::MPS)::Number
  N = length(M1)
  if length(M2) != N
      throw(DimensionMismatch("inner: mismatched lengths $N and $(length(M2))"))
  end
  M1dag = dag(M1)
  simlinks!(M1dag)
  O = M1dag[1]*M2[1]
  for j in eachindex(M1)[2:end]
    O = (O*M1dag[j])*M2[j]
  end
  return O[]
end

function replacebond!(M::MPS,
                      b::Int,
                      phi::ITensor;
                      kwargs...)
  FU,FV,spec = factorize(phi,inds(M[b]); which_factorization="automatic",
                           tags=tags(linkindex(M,b)), kwargs...)
  M[b]   = FU
  M[b+1] = FV

  dir = get(kwargs,:dir,"center")
  if dir=="fromright"
    M.llim_ ==b && (M.llim_ -= 1)
    M.rlim_ == b+2 && (M.rlim_ -= 1)
  elseif dir=="fromleft"
    M.llim_ == b-1 && (M.llim_ += 1)
    M.rlim_ == b+1 && (M.rlim_ += 1)
  end
  return spec
end

"""
    sample!(m::MPS)

Given a normalized MPS m, returns a `Vector{Int}` 
of `length(m)` corresponding to one sample 
of the probability distribution defined by 
squaring the components of the tensor
that the MPS represents. If the MPS does
not have an orthogonality center, 
orthogonalize!(m,1) will be called before
computing the sample.
"""
function sample!(m::MPS)
  orthogonalize!(m,1)
  return sample(m)
end

"""
    sample(m::MPS)

Given a normalized MPS m with `orthoCenter(m)==1`,
returns a `Vector{Int}` of `length(m)`
corresponding to one sample of the
probability distribution defined by 
squaring the components of the tensor
that the MPS represents
"""
function sample(m::MPS)
  N = length(m)

  if orthoCenter(m) != 1
    error("sample: MPS m must have orthoCenter(m)==1")
  end
  if abs(1.0-norm(m[1])) > 1E-8
    error("sample: MPS is not normalized, norm=$(norm(m[1]))")
  end

  result = zeros(Int,N)
  A = m[1]

  for j=1:N
    s = siteindex(m,j)
    d = dim(s)
    # Compute the probability of each state
    # one-by-one and stop when the random
    # number r is below the total prob so far
    pdisc = 0.0
    r = rand()
    # Will need n,An, and pn below
    n = 1
    An = ITensor()
    pn = 0.0
    while n <= d
      projn = ITensor(s)
      projn[s[n]] = 1.0
      An = A*projn
      pn = scalar(dag(An)*An) |> real
      pdisc += pn
      (r < pdisc) && break
      n += 1
    end

    result[j] = n

    if j < N
      A = m[j+1]*An
      A *= (1.0/sqrt(pn))
    end
  end
  return result
end


@doc """
inner(psi::MPS, phi::MPS)

Compute <psi|phi>
""" inner


