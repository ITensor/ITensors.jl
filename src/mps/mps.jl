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
       linkind,
       siteind,
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

function MPS(::Type{T},sites) where {T<:Number}
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  l = [Index(1, "Link,l=$ii") for ii=1:N-1]
  for ii in eachindex(sites)
    s = sites[ii]
    if ii == 1
      v[ii] = ITensor(T,l[ii], s)
    elseif ii == N
      v[ii] = ITensor(T,l[ii-1], s)
    else
      v[ii] = ITensor(T,l[ii-1],s,l[ii])
    end
  end
  return MPS(N,v,0,N+1)
end

MPS(sites) = MPS(Float64,sites)

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

function randomMPS(::Type{T}, sites) where {T<:Number}
  M = MPS(T, sites)
  for i in eachindex(sites)
    randn!(M[i])
    normalize!(M[i])
  end
  M.llim_ = 1
  M.rlim_ = length(M)
  return M
end

randomMPS(sites) = randomMPS(Float64, sites)

function productMPS(::Type{T}, ivals::Vector{<:IndexVal}) where {T<:Number}
  N = length(ivals)
  M = MPS(N)
  if hasqns(ind(ivals[1]))
    links = [Index(QN()=>1;tags="Link,l=$n") for n=1:N]
  else
    links = [Index(1,"Link,l=$n") for n=1:N]
  end
  M[1] = ITensor(T,ind(ivals[1]),links[1])
  M[1][ivals[1],links[1](1)] = 1.0
  for n=2:N-1
    s = ind(ivals[n])
    M[n] = ITensor(T,dag(links[n-1]),s,links[n])
    M[n][links[n-1](1),ivals[n],links[n](1)] = 1.0
  end
  M[N] = ITensor(T,dag(links[N-1]),ind(ivals[N]))
  M[N][links[N-1](1),ivals[N]] = 1.0
  return M
end

productMPS(ivals::Vector{<:IndexVal}) = productMPS(Float64, ivals::Vector{<:IndexVal})

function productMPS(::Type{T}, sites,
                    states) where {T<:Number}
  if length(sites) != length(states)
    throw(DimensionMismatch("Number of sites and and initial states don't match"))
  end
  ivals = [state(sites[n],states[n]) for n=1:length(sites)]
  return productMPS(T, ivals)
end

productMPS(sites, states) = productMPS(Float64, sites, states)

function linkind(M::MPS,j::Integer)
  N = length(M)
  j ≥ length(M) && error("No link index to the right of site $j (length of MPS is $N)")
  li = commonind(M[j],M[j+1])
  if isnothing(li)
    error("linkind: no MPS link index at link $j")
  end
  return li
end

function siteind(M::MPS,j::Integer)
  N = length(M)
  if j == 1
    si = uniqueind(M[j],M[j+1])
  elseif j == N
    si = uniqueind(M[j],M[j-1])
  else
    si = uniqueind(M[j],M[j-1],M[j+1])
  end
  return si
end

function siteinds(M::MPS)
  return [siteind(M,j) for j in 1:length(M)]
end

function replacesites!(M::MPS,sites)
  for j in eachindex(M)
    sj = siteind(M,j)
    replaceind!(M[j],sj,sites[j])
  end
  return M
end

"""
inner(psi::MPS, phi::MPS)

Compute <psi|phi>
"""
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
  dir = get(kwargs, :dir, "fromleft")
  which_decomp = get(kwargs, :which_decomp, "automatic")
  if dir == "fromleft"
    ortho = "left"
  elseif dir == "fromright"
    ortho = "right"
  else
    error("In replacebond!, dir keyword $dir not supported. Use fromleft or fromright")
  end
  kwargs_factorize = Dict(kwargs)
  delete!(kwargs_factorize, :dir)
  L,R,spec = factorize(phi,inds(M[b]); which_decomp = which_decomp,
                                       tags = tags(linkind(M,b)),
                                       ortho = ortho,
                                       kwargs_factorize...)
  M[b]   = L
  M[b+1] = R
  if dir == "fromleft"
    M.llim_ == b-1 && (M.llim_ += 1)
    M.rlim_ == b+1 && (M.rlim_ += 1)
  else
    M.llim_ == b   && (M.llim_ -= 1)
    M.rlim_ == b+2 && (M.rlim_ -= 1)
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
    s = siteind(m,j)
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
