
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

function MPS(A::Vector{<:ITensor}; ortho_lims::UnitRange=1:length(A))
  return MPS(A, first(ortho_lims) - 1, last(ortho_lims) + 1)
end

set_data(A::MPS, data::Vector{ITensor}) = MPS(data, A.llim, A.rlim)

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
function MPS(N::Int; ortho_lims::UnitRange=1:N)
  return MPS(Vector{ITensor}(undef, N); ortho_lims=ortho_lims)
end

"""
    MPS([::Type{ElT} = Float64, ]sites; linkdims=1)

Construct an MPS filled with Empty ITensors of type `ElT` from a collection of indices.

Optionally specify the link dimension with the keyword argument `linkdims`, which by default is 1.

In the future we may generalize `linkdims` to allow specifying each individual link dimension as a vector,
and additionally allow specifying quantum numbers.
"""
function MPS(::Type{T}, sites::Vector{<:Index}; linkdims::Integer=1) where {T<:Number}
  N = length(sites)
  v = Vector{ITensor}(undef, N)
  if N == 1
    v[1] = emptyITensor(T, sites[1])
    return MPS(v)
  end

  space = if hasqns(sites)
    [QN() => linkdims]
  else
    linkdims
  end

  l = [Index(space, "Link,l=$ii") for ii in 1:(N - 1)]
  for ii in eachindex(sites)
    s = sites[ii]
    if ii == 1
      v[ii] = emptyITensor(T, l[ii], s)
    elseif ii == N
      v[ii] = emptyITensor(T, dag(l[ii - 1]), s)
    else
      v[ii] = emptyITensor(T, dag(l[ii - 1]), s, l[ii])
    end
  end
  return MPS(v)
end

MPS(sites::Vector{<:Index}, args...; kwargs...) = MPS(Float64, sites, args...; kwargs...)

function randomU(s1::Index, s2::Index)
  if !hasqns(s1) && !hasqns(s2)
    mdim = dim(s1) * dim(s2)
    RM = randn(mdim, mdim)
    Q, _ = NDTensors.qr_positive(RM)
    G = itensor(Q, dag(s1), dag(s2), s1', s2')
  else
    M = randomITensor(QN(), s1', s2', dag(s1), dag(s2))
    U, S, V = svd(M, (s1', s2'))
    u = commonind(U, S)
    v = commonind(S, V)
    replaceind!(U, u, v)
    G = U * V
  end
  return G
end

function randomizeMPS!(M::MPS, sites::Vector{<:Index}, linkdim=1)
  N = length(sites)
  c = div(N, 2)
  max_pass = 100
  for pass in 1:max_pass, half in 1:2
    if half == 1
      (db, brange) = (+1, 1:1:(N - 1))
    else
      (db, brange) = (-1, N:-1:2)
    end
    for b in brange
      s1 = sites[b]
      s2 = sites[b + db]
      G = randomU(s1, s2)
      T = noprime(G * M[b] * M[b + db])
      rinds = uniqueinds(M[b], M[b + db])
      U, S, V = svd(T, rinds; maxdim=linkdim, utags="Link,l=$(b-1)")
      M[b] = U
      M[b + db] = S * V
      M[b + db] /= norm(M[b + db])
    end
    if half == 2 && dim(commonind(M[c], M[c + 1])) >= linkdim
      break
    end
  end
  setleftlim!(M, 0)
  setrightlim!(M, 2)
  if dim(commonind(M[c], M[c + 1])) < linkdim
    @warn "MPS center bond dimension is less than requested (you requested $linkdim, but in practice it is $(dim(commonind(M[c], M[c + 1]))). This is likely due to technicalities of truncating quantum number sectors."
  end
end

function randomCircuitMPS(
  ::Type{ElT}, sites::Vector{<:Index}, linkdim::Int; kwargs...
) where {ElT<:Number}
  _rmatrix(::Type{Float64}, n, m) = NDTensors.random_orthog(n, m)
  _rmatrix(::Type{ComplexF64}, n, m) = NDTensors.random_unitary(n, m)

  N = length(sites)
  M = MPS(N)

  if N == 1
    M[1] = ITensor(randn(dim(sites[1])), sites[1])
    M[1] /= norm(M[1])
    return M
  end

  l = Vector{Index}(undef, N)

  d = dim(sites[N])
  chi = min(linkdim, d)
  l[N - 1] = Index(chi, "Link,l=$(N-1)")
  O = _rmatrix(ElT, chi, d)
  M[N] = itensor(O, l[N - 1], sites[N])

  for j in (N - 1):-1:2
    chi *= dim(sites[j])
    chi = min(linkdim, chi)
    l[j - 1] = Index(chi, "Link,l=$(j-1)")
    O = _rmatrix(ElT, chi, dim(sites[j]) * dim(l[j]))
    T = reshape(O, (chi, dim(sites[j]), dim(l[j])))
    M[j] = itensor(T, l[j - 1], sites[j], l[j])
  end

  O = _rmatrix(ElT, 1, dim(sites[1]) * dim(l[1]))
  l0 = Index(1, "Link,l=0")
  T = reshape(O, (1, dim(sites[1]), dim(l[1])))
  M[1] = itensor(T, l0, sites[1], l[1])
  M[1] *= onehot(l0 => 1)

  M.llim = 0
  M.rlim = 2

  return M
end

function randomCircuitMPS(sites::Vector{<:Index}, linkdim::Integer; kwargs...)
  return randomCircuitMPS(Float64, sites, linkdim; kwargs...)
end

"""
    randomMPS(::Type{ElT<:Number}, sites::Vector{<:Index}; linkdims=1)

Construct a random MPS with link dimension `linkdims` of 
type `ElT`.
"""
function randomMPS(
  ::Type{ElT}, sites::Vector{<:Index}; linkdims::Integer=1
) where {ElT<:Number}
  if any(hasqns, sites)
    error("initial state required to use randomMPS with QNs")
  end

  # For non-QN-conserving MPS, instantiate
  # the random MPS directly as a circuit:
  return randomCircuitMPS(ElT, sites, linkdims)
end

"""
    randomMPS(sites::Vector{<:Index}; linkdims=1)

Construct a random MPS with link dimension `linkdim` of 
type `Float64`.
"""
function randomMPS(sites::Vector{<:Index}; linkdims::Integer=1)
  return randomMPS(Float64, sites; linkdims=linkdims)
end

function randomMPS(sites::Vector{<:Index}, state; linkdims::Integer=1)
  return randomMPS(Float64, sites, state; linkdims=linkdims)
end

function randomMPS(ElType::Type, sites::Vector{<:Index}, state; linkdims::Integer=1)::MPS
  M = MPS(ElType, sites, state)
  if linkdims > 1
    randomizeMPS!(M, sites, linkdims)
  end
  return M
end

@doc """
    randomMPS(sites::Vector{<:Index}, state; linkdims=1)

Construct a real, random MPS with link dimension `linkdims`,
made by randomizing an initial product state specified by
`state`. This version of `randomMPS` is necessary when creating
QN-conserving random MPS (consisting of QNITensors). The initial
`state` array provided determines the total QN of the resulting
random MPS.
""" randomMPS(::Vector{<:Index}, ::Any)

"""
    MPS(::Type{T<:Number}, ivals::Vector{<:Pair{<:Index}})

Construct a product state MPS with element type `T` and
nonzero values determined from the input IndexVals.
"""
function MPS(::Type{T}, ivals::Vector{<:Pair{<:Index}}) where {T<:Number}
  N = length(ivals)
  M = MPS(N)

  if N == 1
    M[1] = emptyITensor(T, ind(ivals[1]))
    M[1][ivals[1]] = one(T)
    return M
  end

  if hasqns(ind(ivals[1]))
    lflux = QN()
    for j in 1:(N - 1)
      lflux += qn(ivals[j])
    end
    links = Vector{QNIndex}(undef, N - 1)
    for j in (N - 1):-1:1
      links[j] = Index(lflux => 1; tags="Link,l=$j", dir=In)
      lflux -= qn(ivals[j])
    end
  else
    links = [Index(1, "Link,l=$n") for n in 1:(N - 1)]
  end

  M[1] = emptyITensor(T, ind(ivals[1]), links[1])
  M[1][ivals[1], links[1] => 1] = one(T)
  for n in 2:(N - 1)
    s = ind(ivals[n])
    M[n] = emptyITensor(T, dag(links[n - 1]), s, links[n])
    M[n][links[n - 1] => 1, ivals[n], links[n] => 1] = one(T)
  end
  M[N] = emptyITensor(T, dag(links[N - 1]), ind(ivals[N]))
  M[N][links[N - 1] => 1, ivals[N]] = one(T)

  return M
end

# For backwards compatibility
const productMPS = MPS

"""
    MPS(ivals::Vector{<:Pair{<:Index}})

Construct a product state MPS with element type `Float64` and
nonzero values determined from the input IndexVals.
"""
MPS(ivals::Vector{<:Pair{<:Index}}) = MPS(Float64, ivals)

"""
    MPS(::Type{T},
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
psi = MPS(ComplexF64, sites, states)
phi = MPS(sites, "Up")
```
"""
function MPS(::Type{T}, sites::Vector{<:Index}, states_) where {T<:Number}
  if length(sites) != length(states_)
    throw(DimensionMismatch("Number of sites and and initial vals don't match"))
  end
  N = length(states_)
  M = MPS(N)

  if N == 1
    M[1] = state(sites[1], states_[1])
    return M
  end

  states = [state(sites[j], states_[j]) for j in 1:N]

  if hasqns(states[1])
    lflux = QN()
    for j in 1:(N - 1)
      lflux += flux(states[j])
    end
    links = Vector{QNIndex}(undef, N - 1)
    for j in (N - 1):-1:1
      links[j] = Index(lflux => 1; tags="Link,l=$j", dir=In)
      lflux -= flux(states[j])
    end
  else
    links = [Index(1; tags="Link,l=$n") for n in 1:N]
  end

  M[1] = ITensor(T, sites[1], links[1])
  M[1] += states[1] * state(links[1], 1)
  for n in 2:(N - 1)
    M[n] = ITensor(T, dag(links[n - 1]), sites[n], links[n])
    M[n] += state(dag(links[n - 1]), 1) * states[n] * state(links[n], 1)
  end
  M[N] = ITensor(T, dag(links[N - 1]), sites[N])
  M[N] += state(dag(links[N - 1]), 1) * states[N]

  return M
end

function MPS(
  ::Type{T}, sites::Vector{<:Index}, state::Union{String,Integer}
) where {T<:Number}
  return MPS(T, sites, fill(state, length(sites)))
end

function MPS(::Type{T}, sites::Vector{<:Index}, states::Function) where {T<:Number}
  states_vec = [states(n) for n in 1:length(sites)]
  return MPS(T, sites, states_vec)
end

"""
    MPS(sites::Vector{<:Index},states)

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
psi = MPS(sites,states)
```
"""
MPS(sites::Vector{<:Index}, states) = MPS(Float64, sites, states)

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
function replacebond!(M::MPS, b::Int, phi::ITensor; kwargs...)
  ortho::String = get(kwargs, :ortho, "left")
  swapsites::Bool = get(kwargs, :swapsites, false)
  which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
  normalize::Bool = get(kwargs, :normalize, false)

  # Deprecated keywords
  if haskey(kwargs, :dir)
    error(
      """dir keyword in replacebond! has been replaced by ortho.
      Note that the options are now the same as factorize, so use `left` instead of `fromleft` and `right` instead of `fromright`.""",
    )
  end

  indsMb = inds(M[b])
  if swapsites
    sb = siteind(M, b)
    sbp1 = siteind(M, b + 1)
    indsMb = replaceind(indsMb, sb, sbp1)
  end

  L, R, spec = factorize(
    phi, indsMb; which_decomp=which_decomp, tags=tags(linkind(M, b)), kwargs...
  )

  M[b] = L
  M[b + 1] = R
  if ortho == "left"
    leftlim(M) == b - 1 && setleftlim!(M, leftlim(M) + 1)
    rightlim(M) == b + 1 && setrightlim!(M, rightlim(M) + 1)
    normalize && (M[b + 1] ./= norm(M[b + 1]))
  elseif ortho == "right"
    leftlim(M) == b && setleftlim!(M, leftlim(M) - 1)
    rightlim(M) == b + 2 && setrightlim!(M, rightlim(M) - 1)
    normalize && (M[b] ./= norm(M[b]))
  else
    error(
      "In replacebond!, got ortho = $ortho, only currently supports `left` and `right`."
    )
  end
  return spec
end

"""
    replacebond(M::MPS, b::Int, phi::ITensor; kwargs...)

Like `replacebond!`, but returns the new MPS.
"""
function replacebond(M0::MPS, b::Int, phi::ITensor; kwargs...)
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
  if abs(1.0 - norm(m[1])) > 1E-8
    error("sample: MPS is not normalized, norm=$(norm(m[1]))")
  end

  result = zeros(Int, N)
  A = m[1]

  for j in 1:N
    s = siteind(m, j)
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
      projn[s => n] = 1.0
      An = A * dag(projn)
      pn = real(scalar(dag(An) * An))
      pdisc += pn
      (r < pdisc) && break
      n += 1
    end
    result[j] = n
    if j < N
      A = m[j + 1] * An
      A *= (1.0 / sqrt(pn))
    end
  end
  return result
end

_op_prod(o1::AbstractString, o2::AbstractString) = "$o1 * $o2"
_op_prod(o1::Matrix{<:Number}, o2::Matrix{<:Number}) = o1 * o2

"""
    correlation_matrix(psi::MPS,
                       Op1::AbstractString,
                       Op2::AbstractString;
                       kwargs...)

    correlation_matrix(psi::MPS,
                       Op1::Matrix{<:Number},
                       Op2::Matrix{<:Number};
                       kwargs...)

Given an MPS psi and two strings denoting
operators (as recognized by the `op` function), 
computes the two-point correlation function matrix
C[i,j] = <psi| Op1i Op2j |psi>
using efficient MPS techniques. Returns the matrix C.

# Optional Keyword Arguments
- `site_range = 1:length(psi)`: compute correlations only for sites in the given range
- `ishermitian = false` : if `false`, force independent calculations of the matrix elements above and below the diagonal, while if `true` assume they are complex conjugates.

For a correlation matrix of size NxN and an MPS of typical
bond dimension m, the scaling of this algorithm is N^2*m^3.

# Examples
```julia
N = 30
m = 4

s = siteinds("S=1/2",N)
psi = randomMPS(s; linkdims=m)
Czz = correlation_matrix(psi,"Sz","Sz")
Czz = correlation_matrix(psi,[1/2 0; 0 -1/2],[1/2 0; 0 -1/2]) # same as above

s = siteinds("Electron",N; conserve_qns=true)
psi = randomMPS(s, n->isodd(n) ? "Up" : "Dn"; linkdims=m)
Cuu = correlation_matrix(psi,"Cdagup","Cup";site_range=2:8)
``` 
"""
function correlation_matrix(psi::MPS, _Op1, _Op2; kwargs...)
  N = length(psi)
  ElT = promote_itensor_eltype(psi)
  s = siteinds(psi)

  Op1 = _Op1 #make copies into which we can insert "F" string operators, and then restore.
  Op2 = _Op2
  onsiteOp = _op_prod(Op1, Op2)
  fermionic1 = has_fermion_string(Op1, s[1])
  fermionic2 = has_fermion_string(Op2, s[1])
  if fermionic1 != fermionic2
    error(
      "correlation_matrix: Mixed fermionic and bosonic operators are not supported yet."
    )
  end

  # Decide if we need to calculate a non-hermitian corr. matrix which is roughly double the work.
  is_cm_hermitian = false #Assume corr-matrix is non-hermitian
  if haskey(kwargs, :ishermitian) #Did the user explicitly request something?
    is_cm_hermitian::Bool = get(kwargs, :ishermitian, false) #Honour users request
  else
    O1 = op(Op1, s, 1)
    O2 = op(Op2, s, 1)
    O1 /= norm(O1)
    O2 /= norm(O2)
    #We need to decide if O1 ∝ O2 or O1 ∝ O2^dagger allowing for some round off errors.
    eps = 1e-10
    is_op_proportional = norm(O1 - O2) < eps
    is_op_hermitian = norm(O1 - dag(swapprime(O2, 0, 1))) < eps
    if is_op_proportional || is_op_hermitian
      is_cm_hermitian = true
    end
    # finally if they are both fermionic and proportional then the corr matrix will
    # be anti symmetric insterad of Hermitian. Handle things like <C_i*C_j>
    # at this point we know fermionic2=fermionic1, but we put them both in the if
    # to clarify the meaning of what we are doing.
    if is_op_proportional && fermionic1 && fermionic2
      is_cm_hermitian = false
    end
  end

  if haskey(kwargs, :site_range)
    @warn "The `site_range` keyword arg. to `correlation_matrix` is deprecated: use the keyword `sites` instead"
    sites_ = kwargs[:site_range]
  else
    sites_ = get(kwargs, :sites, 1:N)
  end
  sites = (sites_ isa AbstractRange) ? sites_ : collect(sites_)

  start_site = first(sites)
  end_site = last(sites)

  psi = copy(psi)
  orthogonalize!(psi, start_site)
  norm2_psi = norm(psi[start_site])^2

  # Nb = size of block of correlation matrix
  Nb = length(sites)

  C = zeros(ElT, Nb, Nb)

  if start_site == 1
    L = ITensor(1.0)
  else
    lind = commonind(psi[start_site], psi[start_site - 1])
    L = delta(dag(lind), lind')
  end
  pL = start_site - 1

  for (ni, i) in enumerate(sites[1:(end - 1)])
    while pL < i - 1
      pL += 1
      L = (L * psi[pL]) * dag(prime(psi[pL], "Link"))
    end

    Li = L * psi[i]

    # Get j == i diagonal correlations
    rind = commonind(psi[i], psi[i + 1])
    C[ni, ni] =
      scalar((Li * op(onsiteOp, s, i)) * prime(dag(psi[i]), not(rind))) / norm2_psi

    # Get j > i correlations
    if !using_auto_fermion() && fermionic2
      Op1 = "$Op1 * F"
    end

    Li12 = (Li * op(Op1, s, i)) * dag(prime(psi[i]))
    pL12 = i

    for (n, j) in enumerate(sites[(ni + 1):end])
      nj = ni + n

      while pL12 < j - 1
        pL12 += 1
        if !using_auto_fermion() && fermionic2
          Li12 *= op("F", s[pL12]) * dag(prime(psi[pL12]))
        else
          Li12 *= dag(prime(psi[pL12], "Link"))
        end
        Li12 *= psi[pL12]
      end

      lind = commonind(psi[j], Li12)
      Li12 *= psi[j]

      val = (Li12 * op(Op2, s, j)) * dag(prime(prime(psi[j], "Site"), lind))
      C[ni, nj] = scalar(val) / norm2_psi
      if is_cm_hermitian
        C[nj, ni] = conj(C[ni, nj])
      end

      pL12 += 1
      if !using_auto_fermion() && fermionic2
        Li12 *= op("F", s[pL12]) * dag(prime(psi[pL12]))
      else
        Li12 *= dag(prime(psi[pL12], "Link"))
      end
      @assert pL12 == j
    end #for j
    Op1 = _Op1 #"Restore Op1 with no Fs"

    if !is_cm_hermitian #If isHermitian=false the we must calculate the below diag elements explicitly.

      #  Get j < i correlations by swapping the operators
      if !using_auto_fermion() && fermionic1
        Op2 = "$Op2 * F"
      end
      Li21 = (Li * op(Op2, s, i)) * dag(prime(psi[i]))
      pL21 = i
      if !using_auto_fermion() && fermionic1
        Li21 = -Li21 #Required because we swapped fermionic ops, instead of sweeping right to left.
      end

      for (n, j) in enumerate(sites[(ni + 1):end])
        nj = ni + n

        while pL21 < j - 1
          pL21 += 1
          if !using_auto_fermion() && fermionic1
            Li21 *= op("F", s[pL21]) * dag(prime(psi[pL21]))
          else
            Li21 *= dag(prime(psi[pL21], "Link"))
          end
          Li21 *= psi[pL21]
        end

        lind = commonind(psi[j], Li21)
        Li21 *= psi[j]

        val = (Li21 * op(Op1, s, j)) * dag(prime(prime(psi[j], "Site"), lind))
        C[nj, ni] = scalar(val) / norm2_psi

        pL21 += 1
        if !using_auto_fermion() && fermionic1
          Li21 *= op("F", s[pL21]) * dag(prime(psi[pL21]))
        else
          Li21 *= dag(prime(psi[pL21], "Link"))
        end
        @assert pL21 == j
      end #for j
      Op2 = _Op2 #"Restore Op2 with no Fs"
    end #if is_cm_hermitian

    pL += 1
    L = Li * dag(prime(psi[i], "Link"))
  end #for i

  # Get last diagonal element of C
  i = end_site
  while pL < i - 1
    pL += 1
    L = (L * psi[pL]) * dag(prime(psi[pL], "Link"))
  end
  lind = commonind(psi[i], psi[i - 1])
  C[Nb, Nb] =
    scalar(L * psi[i] * op(onsiteOp, s, i) * prime(prime(dag(psi[i]), "Site"), lind)) /
    norm2_psi

  return C
end

"""
    expect(psi::MPS, op::AbstractString...; kwargs...)
    expect(psi::MPS, op::Matrix{<:Number}...; kwargs...)
    expect(psi::MPS, ops; kwargs...)

Given an MPS `psi` and a single operator name, returns
a vector of the expected value of the operator on 
each site of the MPS. 

If multiple operator names are provided, returns a tuple 
of expectation value vectors.

If a container of operator names is provided, returns the
same type of container with names replaced by vectors
of expectation values.

# Optional Keyword Arguments
- `sites = 1:length(psi)`: compute expected values only for sites in the given range

# Examples

```julia
N = 10

s = siteinds("S=1/2",N)
psi = randomMPS(s; linkdims=8)
Z = expect(psi,"Sz") # compute for all sites
Z = expect(psi,"Sz";sites=2:4) # compute for sites 2,3,4
Z3 = expect(psi,"Sz";sites=3)  # compute for site 3 only (output will be a scalar)
XZ = expect(psi,["Sx","Sz"]) # compute Sx and Sz for all sites
Z = expect(psi,[1/2 0; 0 -1/2]) # same as expect(psi,"Sz")

s = siteinds("Electron",N)
psi = randomMPS(s; linkdims=8)
dens = expect(psi,"Ntot")
updens,dndens = expect(psi,"Nup","Ndn") # pass more than one operator
```
"""
function expect(psi::MPS, ops; kwargs...)
  psi = copy(psi)
  N = length(psi)
  ElT = promote_itensor_eltype(psi)
  s = siteinds(psi)

  if haskey(kwargs, :site_range)
    @warn "The `site_range` keyword arg. to `expect` is deprecated: use the keyword `sites` instead"
    sites = kwargs[:site_range]
  else
    sites = get(kwargs, :sites, 1:N)
  end

  site_range = (sites isa AbstractRange) ? sites : collect(sites)
  Ns = length(site_range)
  start_site = first(site_range)

  el_types = map(o -> ishermitian(op(o, s[start_site])) ? real(ElT) : ElT, ops)

  orthogonalize!(psi, start_site)
  norm2_psi = norm(psi)^2

  ex = map((o, el_t) -> zeros(el_t, Ns), ops, el_types)
  for (entry, j) in enumerate(site_range)
    orthogonalize!(psi, j)
    for (n, opname) in enumerate(ops)
      val = scalar(psi[j] * op(opname, s[j]) * dag(prime(psi[j], s[j]))) / norm2_psi
      ex[n][entry] = (el_types[n] <: Real) ? real(val) : val
    end
  end

  if sites isa Number
    return map(arr -> arr[1], ex)
  end
  return ex
end

function expect(psi::MPS, op::AbstractString; kwargs...)
  return first(expect(psi, (op,); kwargs...))
end

function expect(psi::MPS, op::Matrix{<:Number}; kwargs...)
  return first(expect(psi, (op,); kwargs...))
end

function expect(psi::MPS, op1::AbstractString, ops::AbstractString...; kwargs...)
  return expect(psi, (op1, ops...); kwargs...)
end

function expect(psi::MPS, op1::Matrix{<:Number}, ops::Matrix{<:Number}...; kwargs...)
  return expect(psi, (op1, ops...); kwargs...)
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, M::MPS)
  g = create_group(parent, name)
  attributes(g)["type"] = "MPS"
  attributes(g)["version"] = 1
  N = length(M)
  write(g, "length", N)
  write(g, "rlim", M.rlim)
  write(g, "llim", M.llim)
  for n in 1:N
    write(g, "MPS[$(n)]", M[n])
  end
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{MPS})
  g = open_group(parent, name)
  if read(attributes(g)["type"]) != "MPS"
    error("HDF5 group or file does not contain MPS data")
  end
  N = read(g, "length")
  rlim = read(g, "rlim")
  llim = read(g, "llim")
  v = [read(g, "MPS[$(i)]", ITensor) for i in 1:N]
  return MPS(v, llim, rlim)
end
