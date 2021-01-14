
"""
    MPS

A finite size matrix product state type.
Keeps track of the orthogonality center.
"""
mutable struct MPS <: AbstractMPS
  data::Vector{ITensor}
  llim::Int
  rlim::Int
end

function MPS(A::Vector{<:ITensor};
             ortho_lims::UnitRange = 1:length(A))
  return MPS(A, first(ortho_lims)-1, last(ortho_lims)+1)
end

@doc """
    MPS(v::Vector{<:ITensor})

Construct an MPS from a Vector of ITensors.
""" MPS(v::Vector{<:ITensor})

"""
    MPS()

Construct an empty MPS with zero sites.
"""
MPS() = MPS(ITensor[], 0, 0)

"""
    MPS(N::Int)

Construct an MPS with N sites with default constructed
ITensors.
"""
MPS(N::Int; ortho_lims::UnitRange = 1:N) =
  MPS(Vector{ITensor}(undef, N); ortho_lims = ortho_lims)

"""
    MPS([::Type{ElT} = Float64, ]sites)

Construct an MPS filled with Empty ITensors of type `ElT` from a collection of indices.
"""
function MPS(::Type{T}, sites::Vector{<:Index}) where {T <: Number}
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  if N==1
    v[1] = emptyITensor(T,sites[1])
    return MPS(v)
  end

  l = [Index(1, "Link,l=$ii") for ii=1:N-1]
  for ii in eachindex(sites)
    s = sites[ii]
    if ii == 1
      v[ii] = emptyITensor(T,l[ii], s)
    elseif ii == N
      v[ii] = emptyITensor(T,l[ii-1], s)
    else
      v[ii] = emptyITensor(T,l[ii-1],s,l[ii])
    end
  end
  return MPS(v)
end

MPS(sites::Vector{<:Index}) = MPS(Float64, sites)

function randomU(s1::Index, s2::Index)
  if !hasqns(s1) && !hasqns(s2)
    mdim = dim(s1)*dim(s2)
    RM = randn(mdim,mdim)
    Q,_ = NDTensors.qr_positive(RM)
    G = itensor(Q,dag(s1),dag(s2),s1',s2')
  else
    M = randomITensor(QN(),s1',s2',dag(s1),dag(s2))
    U, S, V = svd(M,(s1',s2'))
    u = commonind(U,S)
    v = commonind(S,V)
    replaceind!(U, u, v)
    G = U * V
  end
  return G
end

function randomizeMPS!(M::MPS, sites::Vector{<:Index}, linkdim=1)
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
      U,S,V = svd(T,rinds;maxdim=linkdim, utags = "Link,l=$(b-1)")
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

function randomCircuitMPS(::Type{Float64},
                          sites::Vector{<:Index},
                          linkdim::Int;
                          kwargs...)
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

function randomCircuitMPS(sites::Vector{<:Index},
                          linkdim::Int;
                          kwargs...)
  return randomCircuitMPS(Float64, sites, linkdim; kwargs...)
end

"""
    randomMPS(::Type{ElT<:Number}, sites::Vector{<:Index}; linkdim=1)

Construct a random MPS with link dimension `linkdim` of 
type `ElT`.
"""
function randomMPS(::Type{ElT},
                   sites::Vector{<:Index},
                   linkdim::Int=1) where {ElT<:Number}
  if hasqns(sites[1])
    error("initial state required to use randomMPS with QNs")
  end

  # For non-QN-conserving MPS, instantiate
  # the random MPS directly as a circuit:
  return randomCircuitMPS(ElT, sites, linkdim)
end

"""
    randomMPS(sites::Vector{<:Index}; linkdim=1)

Construct a random MPS with link dimension `linkdim` of 
type `Float64`.
"""
randomMPS(sites::Vector{<:Index},
          linkdim::Int=1) = randomMPS(Float64, sites, linkdim)

"""
    randomMPS(sites::Vector{<:Index}, state; linkdim=1)

Construct a real, random MPS with link dimension `linkdim`,
made by randomizing an initial product state specified by
`state`. This version of `randomMPS` is necessary when creating
QN-conserving random MPS (consisting of QNITensors). The initial
`state` array provided determines the total QN of the resulting
random MPS.
"""
function randomMPS(sites::Vector{<:Index},
                   state,
                   linkdim::Int=1)::MPS
  M = productMPS(sites, state)
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

  if N==1
    M[1] = emptyITensor(T,ind(ivals[1]))
    M[1][ivals[1]] = one(T)
    return M
  end

  if hasqns(ind(ivals[1]))
    links = [Index(QN()=>1;tags="Link,l=$n") for n=1:N]
  else
    links = [Index(1,"Link,l=$n") for n=1:N]
  end
  M[1] = emptyITensor(T,ind(ivals[1]), links[1])
  M[1][ivals[1],links[1](1)] = one(T)
  for n=2:N-1
    s = ind(ivals[n])
    M[n] = emptyITensor(T,dag(links[n-1]),s,links[n])
    M[n][links[n-1](1),ivals[n],links[n](1)] = one(T)
  end
  M[N] = emptyITensor(T,dag(links[N-1]),ind(ivals[N]))
  M[N][links[N-1](1),ivals[N]] = one(T)
  return M
end

"""
    productMPS(ivals::Vector{<:IndexVal})

Construct a product state MPS with element type `Float64` and
nonzero values determined from the input IndexVals.
"""
productMPS(ivals::Vector{<:IndexVal}) =
  productMPS(Float64,
             ivals::Vector{<:IndexVal})

"""
    productMPS(::Type{T},
               sites::Vector{<:Index},
               states::Union{Vector{String},
                             Vector{Int},
                             String,
                             Int})

Construct a product state MPS of element type `T`, having
site indices `sites`, and which corresponds to the initial
state given by the array `states`. The input `states` may
be an array of strings or an array of ints recognized by the 
`state` function defined for the relevant Index tag type.
In addition, a single string or int can be input to create
a uniform state.

# Examples
```julia
N = 10
sites = siteinds("S=1/2", N)
states = [isodd(n) ? "Up" : "Dn" for n=1:N]
psi = productMPS(ComplexF64, sites, states)
phi = productMPS(sites, "Up")
```
"""
function productMPS(::Type{T},
                    sites::Vector{ <: Index},
                    states) where {T <: Number}
  if length(sites) != length(states)
    throw(DimensionMismatch("Number of sites and and initial states don't match"))
  end
  ivals = [state(sites[n],states[n]) for n=1:length(sites)]
  return productMPS(T, ivals)
end

function productMPS(::Type{T},
                    sites::Vector{ <: Index},
                    states::Union{String, Int}) where {T <: Number}
  ivals = [state(sites[n], states) for n in 1:length(sites)]
  return productMPS(T, ivals)
end

function productMPS(::Type{T},
                    sites::Vector{ <: Index},
                    states::Function) where {T <: Number}
  ivals = [state(sites[n], states(n)) for n in 1:length(sites)]
  return productMPS(T, ivals)
end

"""
    productMPS(sites::Vector{<:Index},states)

Construct a product state MPS having
site indices `sites`, and which corresponds to the initial
state given by the array `states`. The `states` array may
consist of either an array of integers or strings, as 
recognized by the `state` function defined for the relevant
Index tag type.

# Examples
```julia
N = 10
sites = siteinds("S=1/2",N)
states = [isodd(n) ? "Up" : "Dn" for n=1:N]
psi = productMPS(sites,states)
```
"""
productMPS(sites::Vector{ <: Index},
           states) =
  productMPS(Float64, sites, states)

"""
    siteind(M::MPS, j::Int; kwargs...)

Get the first site Index of the MPS. Return `nothing` if none is found.
"""
siteind(M::MPS, j::Int; kwargs...) = siteind(first, M, j; kwargs...)

"""
    siteind(::typeof(only), M::MPS, j::Int; kwargs...)

Get the only site Index of the MPS. Return `nothing` if none is found.
"""
function siteind(::typeof(only), M::MPS, j::Int; kwargs...)
  is = siteinds(M, j; kwargs...)
  if isempty(is)
    return nothing
  end
  return only(is)
end

"""
    siteinds(M::MPS)
    siteinds(::typeof(first), M::MPS)

Get a vector of the first site Index found on each tensor of the MPS.

    siteinds(::typeof(only), M::MPS)

Get a vector of the only site Index found on each tensor of the MPS. Errors if more than one is found.

    siteinds(::typeof(all), M::MPS)

Get a vector of the all site Indices found on each tensor of the MPS. Returns a Vector of IndexSets.
"""
siteinds(M::MPS; kwargs...) = siteinds(first, M; kwargs...)

function replace_siteinds!(M::MPS, sites)
  for j in eachindex(M)
    sj = siteind(only, M, j)
    replaceind!(M[j], sj, sites[j])
  end
  return M
end

replace_siteinds(M::MPS, sites) = replace_siteinds!(copy(M), sites)

"""
    replacebond!(M::MPS, b::Int, phi::ITensor; kwargs...)

Factorize the ITensor `phi` and replace the ITensors
`b` and `b+1` of MPS `M` with the factors. Choose
the orthogonality with `ortho="left"/"right"`.
"""
function replacebond!(M::MPS, b::Int, phi::ITensor;
                      kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  swapsites::Bool = get(kwargs, :swapsites, false)
  which_decomp::Union{String, Nothing} = get(kwargs, :which_decomp, nothing)
  normalize::Bool = get(kwargs, :normalize, false)

  # Deprecated keywords
  if haskey(kwargs, :dir)
    error("""dir keyword in replacebond! has been replaced by ortho.
          Note that the options are now the same as factorize, so use `left` instead of `fromleft` and `right` instead of `fromright`.""")
  end

  indsMb = inds(M[b])
  if swapsites
    sb = siteind(M, b)
    sbp1 = siteind(M, b+1)
    indsMb = replaceind(indsMb, sb, sbp1)
  end

  L,R,spec = factorize(phi, indsMb;
                       which_decomp = which_decomp,
                       tags = tags(linkind(M, b)),
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
    replacebond(M::MPS, b::Int, phi::ITensor; kwargs...)

Like `replacebond!`, but returns the new MPS.
"""
function replacebond(M0::MPS, b::Int, phi::ITensor;
                     kwargs...)
  M = copy(M0)
  replacebond!(M, b, phi; kwargs...)
  return M
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
      An = A * dag(projn)
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

function HDF5.write(parent::Union{HDF5.File,HDF5.Group},
                    name::AbstractString,
                    M::MPS)
  g = create_group(parent,name)
  attributes(g)["type"] = "MPS"
  attributes(g)["version"] = 1
  N = length(M)
  write(g, "length", N)
  write(g, "rlim", M.rlim)
  write(g, "llim", M.llim)
  for n=1:N
    write(g,"MPS[$(n)]", M[n])
  end
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group},
                   name::AbstractString,
                   ::Type{MPS})
  g = open_group(parent,name)
  if read(attributes(g)["type"]) != "MPS"
    error("HDF5 group or file does not contain MPS data")
  end
  N = read(g, "length")
  rlim = read(g, "rlim")
  llim = read(g, "llim")
  v = [read(g,"MPS[$(i)]",ITensor) for i in 1:N]
  return MPS(v, llim, rlim)
end
