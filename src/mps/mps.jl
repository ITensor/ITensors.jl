
"""
    MPS

A finite size matrix product state type.
Keeps track of the orthogonality center.
"""
mutable struct MPS <: AbstractMPS
  length::Int
  data::Vector{ITensor}
  llim::Int
  rlim::Int
  function MPS(N::Int,
               A::Vector{<:ITensor},
               llim::Int = 0,
               rlim::Int = N+1)
    new(N, A, llim, rlim)
  end
end

"""
    MPS()

Construct an empty MPS with zero sites.
"""
MPS() = MPS(0, Vector{ITensor}(), 0, 0)

"""
    MPS(N::Int)

Construct an MPS with N sites with default constructed
ITensors.
"""
MPS(N::Int) = MPS(N, Vector{ITensor}(undef, N))

"""
    MPS(::Type{T<:Number}, sites)

Construct an MPS from a collection of indices
with element type T.
"""
function MPS(::Type{T}, sites) where {T<:Number}
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
  return MPS(N, v)
end

"""
    MPS(sites)

Construct an MPS from a collection of indices
with element type Float64.
"""
MPS(sites) = MPS(Float64, sites)

"""
    MPS(v::Vector{<:ITensor})

Construct an MPS from a Vector of ITensors.
"""
MPS(v::Vector{<:ITensor}) = MPS(length(v), v)

function randomU(s1,s2)
  if !hasqns(s1) && !hasqns(s2)
    mdim = dim(s1)*dim(s2)
    RM = randn(mdim,mdim)
    Q,_ = NDTensors.qr_positive(RM)
    G = itensor(Q,dag(s1),dag(s2),s1',s2')
  else
    M = randomITensor(QN(),s1',s2',dag(s1),dag(s2))
    U,S,V = svd(M,(s1',s2'))
    u = commonind(U,S)
    v = commonind(S,V)
    G = (U*delta(dag(u),v))*V
  end
  return G
end

function randomizeMPS!(M::MPS, sites, linkdim=1)
  N = length(sites)
  c = div(N,2)
  max_pass = 100
  for pass=1:max_pass,half=1:2
    if half==1
      (db,brange) = (+1, 1:1:N-1)
    else
      (db,brange) = (-1, N:-1:2)
    end
    for b=brange
      s1 = sites[b]
      s2 = sites[b+db]
      G = randomU(s1,s2)
      T = noprime(G*M[b]*M[b+db])
      rinds = uniqueinds(M[b],M[b+db])
      U,S,V = svd(T,rinds;maxdim=linkdim)
      M[b] = U
      M[b+db] = S*V
      M[b+db] /= norm(M[b+db])
    end
    if half==2 && dim(commonind(M[c],M[c+1])) >= linkdim
      break
    end
  end
  setleftlim!(M, 0)
  setrightlim!(M, 2)
  if dim(commonind(M[c],M[c+1])) < linkdim
    error("MPS center bond dim less than requested")
  end
end

function randomCircuitMPS(sites,linkdim::Int;kwargs...)::MPS
  N = length(sites)
  M = MPS(N)

  if N==1
    M[1] = ITensor(randn(dim(sites[1])),sites[1])
    M[1] /= norm(M[1])
    return M
  end

  l = Vector{Index}(undef,N)
  
  d = dim(sites[N])
  chi = min(linkdim,d)
  l[N-1] = Index(chi,"Link,l=$(N-1)")
  O = NDTensors.random_orthog(chi,d)
  M[N] = itensor(O,l[N-1],sites[N])

  for j=N-1:-1:2
    chi *= dim(sites[j])
    chi = min(linkdim,chi)
    l[j-1] = Index(chi,"Link,l=$(j-1)")
    O = NDTensors.random_orthog(chi,dim(sites[j])*dim(l[j]))
    T = reshape(O,(chi,dim(sites[j]),dim(l[j])))
    M[j] = itensor(T,l[j-1],sites[j],l[j])
  end

  O = NDTensors.random_orthog(1,dim(sites[1])*dim(l[1]))
  l0 = Index(1,"Link,l=0")
  T = reshape(O,(1,dim(sites[1]),dim(l[1])))
  M[1] = itensor(T,l0,sites[1],l[1])
  M[1] *= setelt(l0=>1)

  M.llim = 0
  M.rlim = 2

  return M
end

"""
    randomMPS(::Type{T<:Number}, sites; linkdim=1)

Construct a random MPS with link dimension `linkdim` of 
type `T`.
"""
function randomMPS(::Type{T}, sites, linkdim::Int=1) where {T<:Number}
  if hasqns(sites[1])
    error("initial state required to use randomMPS with QNs")
  end

  # For non-QN-conserving MPS, instantiate
  # the random MPS directly as a circuit:
  return randomCircuitMPS(sites,linkdim)
end

"""
    randomMPS(sites; linkdim=1)

Construct a random MPS with link dimension `linkdim` of 
type `Float64`.
"""
randomMPS(sites, linkdim::Int=1) = randomMPS(Float64, sites, linkdim)

"""
    randomMPS(sites,state; linkdim=1)

Construct a real, random MPS with link dimension `linkdim`,
made by randomizing an initial product state specified by
`state`.
"""
function randomMPS(sites,state,linkdim::Int=1)::MPS
  M = productMPS(sites,state)
  if linkdim > 1
    randomizeMPS!(M,sites,linkdim)
  end
  return M
end

"""
    productMPS(::Type{T<:Number}, ivals::Vector{<:IndexVal})

Construct a product state MPS with element type `T` and
nonzero values determined from the input IndexVals.
"""
function productMPS(::Type{T},
                    ivals::Vector{<:IndexVal}) where {T<:Number}
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

"""
    productMPS(ivals::Vector{<:IndexVal})

Construct a product state MPS with element type `Float64` and
nonzero values determined from the input IndexVals.
"""
productMPS(ivals::Vector{<:IndexVal}) = productMPS(Float64,
                                                   ivals::Vector{<:IndexVal})

function productMPS(::Type{T},
                    sites,
                    states) where {T<:Number}
  if length(sites) != length(states)
    throw(DimensionMismatch("Number of sites and and initial states don't match"))
  end
  ivals = [state(sites[n],states[n]) for n=1:length(sites)]
  return productMPS(T, ivals)
end

productMPS(sites, states) = productMPS(Float64, sites, states)

function siteind(M::MPS, j::Int)
  N = length(M)
  if j == 1
    si = uniqueind(M[j], M[j+1])
  elseif j == N
    si = uniqueind(M[j], M[j-1])
  else
    si = uniqueind(M[j], M[j-1], M[j+1])
  end
  return si
end

function siteinds(M::MPS)
  return [siteind(M, j) for j in 1:length(M)]
end

function replace_siteinds!(M::MPS, sites)
  for j in eachindex(M)
    sj = siteind(M, j)
    replaceind!(M[j], sj, sites[j])
  end
  return M
end

replace_siteinds(M::MPS, sites) = replace_siteinds!(copy(M), sites)

"""
    dot(psi::MPS, phi::MPS; make_inds_match = true)
    inner(psi::MPS, phi::MPS; make_inds_match = true)

Compute <psi|phi>.

If `make_inds_match = true`, the function attempts to make
the site indices match before contracting (so for example, the
inputs can have different site indices, as long as they 
have the same dimensions or QN blocks).
"""
function LinearAlgebra.dot(M1::MPS, M2::MPS; make_inds_match::Bool = true)::Number
  N = length(M1)
  if length(M2) != N
    throw(DimensionMismatch("inner: mismatched lengths $N and $(length(M2))"))
  end
  M1dag = dag(M1)
  sim_linkinds!(M1dag)
  if make_inds_match
    replace_siteinds!(M1dag, siteinds(M2))
  end
  O = M1dag[1] * M2[1]
  for j in eachindex(M1)[2:end]
    O = (O*M1dag[j])*M2[j]
  end
  return O[]
end

inner(M1::MPS, M2::MPS; kwargs...) = dot(M1, M2; kwargs...)

"""
    replacebond!(M::MPS, b::Int, phi::ITensor; kwargs...)

Factorize the ITensor `phi` and replace the ITensors
`b` and `b+1` of MPS `M` with the factors. Choose
the orthogonality with `ortho="left"/"right"`.
"""
function replacebond!(M::MPS,
                      b::Int,
                      phi::ITensor;
                      kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  which_decomp::Union{String, Nothing} = get(kwargs, :which_decomp, nothing)
  normalize::Bool = get(kwargs, :normalize, false)

  # Deprecated keywords
  if haskey(kwargs, :dir)
    error("""dir keyword in replacebond! has been replaced by ortho.
          Note that the options are now the same as factorize, so use `left` instead of `fromleft` and `right` instead of `fromright`.""")
  end

  L,R,spec = factorize(phi,inds(M[b]); which_decomp = which_decomp,
                                       tags = tags(linkind(M,b)),
                                       kwargs...)
  M[b]   = L
  M[b+1] = R
  if ortho == "left"
    leftlim(M) == b-1 && setleftlim!(M, leftlim(M)+1)
    rightlim(M) == b+1 && setrightlim!(M, rightlim(M)+1)
    normalize && (M[b+1] ./= norm(M[b+1]))
  elseif ortho == "right"
    leftlim(M) == b && setleftlim!(M, leftlim(M)-1)
    rightlim(M) == b+2 && setrightlim!(M, rightlim(M)-1)
    normalize && (M[b] ./= norm(M[b]))
  else
    error("In replacebond!, got ortho = $ortho, only currently supports `left` and `right`.")
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
  orthogonalize!(m, 1)
  return sample(m)
end

"""
    sample(m::MPS)

Given a normalized MPS m with `orthocenter(m)==1`,
returns a `Vector{Int}` of `length(m)`
corresponding to one sample of the
probability distribution defined by
squaring the components of the tensor
that the MPS represents
"""
function sample(m::MPS)
  N = length(m)

  if orthocenter(m) != 1
    error("sample: MPS m must have orthocenter(m)==1")
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
      pn = real(scalar(dag(An)*An))
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

