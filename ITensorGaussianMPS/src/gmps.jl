import Base: sortperm, size, length, eltype, conj, transpose, copy, *
using ITensors: alias
abstract type AbstractSymmetry end
struct ConservesNfParity{T} <: AbstractSymmetry
  data::T
end

struct ConservesNf{T} <: AbstractSymmetry
  data::T
end
#
# Single particle von Neumann entanglement entropy
#
function entropy(n::Number)
  (n ≤ 0 || n ≥ 1) && return 0
  return -(n * log(n) + (1 - n) * log(1 - n))
end

entropy(ns::Vector{Float64}) = sum(entropy, ns)

#
# Linear Algebra tools
#

"""
    frobenius_distance(M1::AbstractMatrix, M2::AbstractMatrix)

Computes the Frobenius distance `√tr((M1-M2)'*(M1-M2))`.
"""
function frobenius_distance(M1::AbstractMatrix, M2::AbstractMatrix)
  return sqrt(abs(tr(M1'M1) + tr(M2'M2) - tr(M1'M2) - tr(M2'M1)))
end

#
# Rotations
#

struct Circuit{T} <: LinearAlgebra.AbstractRotation{T}
  rotations::Vector{Givens{T}}
end

Base.adjoint(R::Circuit) = Adjoint(R)

function Base.show(io::IO, ::MIME"text/plain", C::Circuit{T}) where {T}
  print(io, "Circuit{$T}:\n")
  return show(io, "text/plain", C.rotations)
end

function Base.copy(aR::Adjoint{<:Any,Circuit{T}}) where {T}
  return Circuit{T}(reverse!([r' for r in aR.parent.rotations]))
end

function LinearAlgebra.lmul!(G::Givens, R::Circuit)
  push!(R.rotations, G)
  return R
end

function LinearAlgebra.lmul!(R::Circuit, A::AbstractArray)
  @inbounds for i in 1:length(R.rotations)
    lmul!(R.rotations[i], A)
  end
  return A
end

function LinearAlgebra.rmul!(A::AbstractMatrix, adjR::Adjoint{<:Any,<:Circuit})
  R = adjR.parent
  @inbounds for i in 1:length(R.rotations)
    rmul!(A, adjoint(R.rotations[i]))
  end
  return A
end

Base.:*(g1::Circuit, g2::Circuit) = Circuit(vcat(g2.rotations, g1.rotations))
LinearAlgebra.lmul!(g1::Circuit, g2::Circuit) = append!(g2.rotations, g1.rotations)

Base.:*(A::Circuit, B::Union{<:Hermitian,<:Diagonal}) = A * convert(Matrix, B)
Base.:*(A::Adjoint{<:Any,<:Circuit}, B::Hermitian) = copy(A) * convert(Matrix, B)
Base.:*(A::Adjoint{<:Any,<:Circuit}, B::Diagonal) = copy(A) * convert(Matrix, B)
function Base.:*(A::Adjoint{<:Any,<:AbstractVector}, B::Adjoint{<:Any,<:Circuit})
  return convert(Matrix, A) * B
end

function LinearAlgebra.rmul!(A::AbstractMatrix, R::Circuit)
  @inbounds for i in reverse(1:length(R.rotations))
    rmul!(A, R.rotations[i])
  end
  return A
end

function Base.:*(A::AbstractMatrix, B::Adjoint{<:Any,<:Circuit})
  AB = copy(A)
  rmul!(AB, B)
  return AB
end

function replace!(f, G::Circuit)
  for i in eachindex(G.rotations)
    G.rotations[i] = f(G.rotations[i])
  end
  return G
end

function replace_indices!(f, G::Circuit)
  return replace!(g -> Givens(f(g.i1), f(g.i2), g.c, g.s), G)
end

function shift!(G::Circuit, i::Int)
  return replace_indices!(j -> j + i, G)
end

function scale!(G::Circuit, i::Int)
  return replace_indices!(j -> j * i, G)
end

function conj!(G::Circuit)
  return replace!(g -> Givens(g.i1, g.i2, g.c, g.s'), G)
end

ngates(G::Circuit) = length(G.rotations)

#
# Free fermion tools
#

is_creation_operator(o::Op) = is_creation_operator(ITensors.name(o))
is_creation_operator(o::String) = is_creation_operator(OpName(o))
is_creation_operator(::OpName) = false
is_creation_operator(::OpName"Cdag") = true
is_creation_operator(::OpName"Cdagup") = true
is_creation_operator(::OpName"Cdagdn") = true
is_creation_operator(::OpName"c†") = true
is_creation_operator(::OpName"c†↑") = true
is_creation_operator(::OpName"c†↓") = true

is_annihilation_operator(o::Op) = is_annihilation_operator(ITensors.name(o))
is_annihilation_operator(o::String) = is_annihilation_operator(OpName(o))
is_annihilation_operator(::OpName) = false
is_annihilation_operator(::OpName"C") = true
is_annihilation_operator(::OpName"Cup") = true
is_annihilation_operator(::OpName"Cdn") = true
is_annihilation_operator(::OpName"c") = true
is_annihilation_operator(::OpName"c↑") = true
is_annihilation_operator(::OpName"c↓") = true

expand_to_ladder_operators(o::Op) = expand_to_ladder_operators(ITensors.name(o))
expand_to_ladder_operators(o::String) = expand_to_ladder_operators(OpName(o))
expand_to_ladder_operators(opname::OpName) = opname # By default does nothing
expand_to_ladder_operators(::OpName"N") = ["Cdag", "C"]
expand_to_ladder_operators(::OpName"Nup") = ["Cdagup", "Cup"]
expand_to_ladder_operators(::OpName"Ndn") = ["Cdagdn", "Cdn"]
expand_to_ladder_operators(opname::OpName"n↑") = expand_to_ladder_operators(alias(opname))
expand_to_ladder_operators(opname::OpName"n↓") = expand_to_ladder_operators(alias(opname))

#interlaced_hamiltonian(h::AbstractMatrix) = h
#blocked_hamiltonian(h::AbstractMatrix) = Hermitian(reverse_interleave(Matrix(h)))

function quadrant(term)
  if is_creation_operator(term[1]) && is_annihilation_operator(term[2])
    q = (2, 2)
  elseif is_annihilation_operator(term[1]) && is_creation_operator(term[2])
    q = (1, 1)
  elseif is_annihilation_operator(term[1]) && is_annihilation_operator(term[2])
    q = (1, 2)
  elseif is_creation_operator(term[1]) && is_creation_operator(term[2])
    q = (2, 1)
  else
    error("Unknown quadratic hopping term: $term")
  end
  return q
end

function single_to_quadratic(term)
  site = ITensors.site(term[1])
  new_ops = expand_to_ladder_operators(term[1])
  return coefficient(term) * Op(new_ops[1], site) * Op(new_ops[2], site)
end

function quadratic_operator(os::OpSum)
  os = deepcopy(os)
  #os = ITensors.sorteachterm(os, sites)
  os = ITensors.sortmergeterms(os)

  nterms = length(os)
  coefs = Vector{Number}(undef, nterms)
  sites = Vector{Tuple{Int,Int}}(undef, nterms)
  quads = Vector{Tuple{Int,Int}}(undef, nterms)
  nsites = 0
  # detect terms and size of lattice
  for n in 1:nterms
    term = os[n]
    #@show term
    #@show term.coef
    coef = isreal(coefficient(term)) ? real(coefficient(term)) : coefficient(term)
    coefs[n] = coef
    term = (length(term) == 1) ? single_to_quadratic(term) : term
    length(term) ≠ 2 && error("Must create hopping Hamiltonian from quadratic Hamiltonian")
    quads[n] = quadrant(term)
    sites[n] = ntuple(n -> ITensors.site(term[n]), Val(2))
    nsites = max(nsites, maximum(sites[n]))
  end
  # detect coefficient type 
  coef_type = mapreduce(typeof, promote_type, coefs)
  ElT = isreal(coefs) ? real(coef_type) : coef_type
  # fill Hamiltonian matrix with elements
  h = zeros(ElT, 2 * nsites, 2 * nsites)
  other_quad = i -> i == 2 ? 1 : 2
  for n in 1:nterms
    quad = quads[n]
    offsets = nsites .* (quad .- 1)
    if quad[1] != quad[2]
      h[(sites[n] .+ offsets)...] += coefs[n]
    else
      h[(sites[n] .+ offsets)...] += 0.5 * coefs[n]
      other_offsets = nsites .* (other_quad.(quad) .- 1)
      h[(sites[n] .+ other_offsets)...] += -0.5 * conj(coefs[n])
    end
  end
  return interleave(h)
end

function quadratic_operator(os_up::OpSum, os_dn::OpSum)
  h_up = quadratic_operator(os_up)
  h_dn = quadratic_operator(os_dn)
  @assert size(h_up) == size(h_dn)
  N = size(h_up, 1)
  h = zeros(eltype(h_up), (2 * N, 2 * N))
  n = div(N, 2)
  # interlace the blocks of both quadratic hamiltonians
  h_up = reverse_interleave(Matrix(h_up))
  h_dn = reverse_interleave(Matrix(h_dn))
  # super-quadrant (1,1)
  h[1:2:N, 1:2:N] = h_up[1:n, 1:n]
  h[2:2:N, 2:2:N] = h_dn[1:n, 1:n]
  # super-quadrant (2,1)
  h[(N + 1):2:(2 * N), 1:2:N] = h_up[(n + 1):(2 * n), 1:n]
  h[(N + 2):2:(2 * N), 2:2:N] = h_dn[(n + 1):(2 * n), 1:n]
  # super-quadrant (2,2)
  h[(N + 1):2:(2 * N), (N + 1):2:(2 * N)] = h_up[(n + 1):N, (n + 1):N]
  h[(N + 2):2:(2 * N), (N + 2):2:(2 * N)] = h_dn[(n + 1):N, (n + 1):N]
  # super-quadrant (1,2)
  h[1:2:N, (N + 1):2:(2 * N)] = h_up[1:n, (n + 1):(2 * n)]
  h[2:2:N, (N + 2):2:(2 * N)] = h_dn[1:n, (n + 1):(2 * n)]
  #convert from blocked to interlaced format. Odd base-rows are spin-up, even are spin-down.
  return interleave(h)
end

quadratic_hamiltonian(os::OpSum) = Hermitian(quadratic_operator(os))
function quadratic_hamiltonian(os_up::OpSum, os_dn::OpSum)
  return Hermitian(quadratic_operator(os_up, os_dn))
end

function hopping_operator(os::OpSum; drop_pairing_terms_tol=nothing)
  # convert to blocked format
  h = reverse_interleave(Matrix(quadratic_hamiltonian(os)))
  # check that offdiagonal blocks are 0
  N = div(size(h, 1), 2)
  if isnothing(drop_pairing_terms_tol)
    drop_pairing_terms_tol = eps(real(eltype(h)))
  end
  if !all(abs.(h[1:N, (N + 1):(2 * N)]) .< drop_pairing_terms_tol)
    error("Trying to convert hamiltonian with pairing terms to hopping hamiltonian!")
  end
  return 2 .* h[(N + 1):(2 * N), (N + 1):(2 * N)]
end

# Make a combined hopping Hamiltonian for spin up and down
function hopping_operator(os_up::OpSum, os_dn::OpSum; drop_pairing_terms_tol=nothing)
  # convert to blocked format
  h = reverse_interleave(Matrix(quadratic_hamiltonian(os_up, os_dn)))
  # check that offdiagonal blocks are 0
  N = div(size(h, 1), 2)
  if isnothing(drop_pairing_terms_tol)
    drop_pairing_terms_tol = eps(real(eltype(h)))
  end
  if !all(abs.(h[1:N, (N + 1):(2 * N)]) .< drop_pairing_terms_tol)
    error("Trying to convert hamiltonian with pairing terms to hopping hamiltonian!")
  end
  return 2 .* h[(N + 1):(2 * N), (N + 1):(2 * N)]
end

function hopping_hamiltonian(os::OpSum; drop_pairing_terms_tol=nothing)
  return Hermitian(hopping_operator(os; drop_pairing_terms_tol))
end
function hopping_hamiltonian(os_up::OpSum, os_dn::OpSum; drop_pairing_terms_tol=nothing)
  return Hermitian(hopping_operator(os_up, os_dn; drop_pairing_terms_tol))
end

function slater_determinant_matrix(h::AbstractMatrix, Nf::Int)
  _, u = eigen(h)
  return u[:, 1:Nf]
end

#
# Correlation matrix diagonalization
#

struct Boguliobov
  u::Givens
end

set_data(::ConservesNf, x) = ConservesNf(x)
set_data(::ConservesNfParity, x) = ConservesNfParity(x)
site_stride(::ConservesNf) = 1
site_stride(::ConservesNfParity) = 2
copy(A::T) where {T<:AbstractSymmetry} = T(copy(A.data))
size(A::T) where {T<:AbstractSymmetry} = size(A.data)
size(A::T, dim::Int) where {T<:AbstractSymmetry} = size(A.data, dim)

length(A::T) where {T<:AbstractSymmetry} = length(A.data)
eltype(A::T) where {T<:AbstractSymmetry} = eltype(A.data)
Hermitian(A::T) where {T<:AbstractSymmetry} = set_data(A, Hermitian(A.data))
conj(A::T) where {T<:AbstractSymmetry} = set_data(A, conj(A.data))
transpose(A::T) where {T<:AbstractSymmetry} = set_data(A, transpose(A.data))

"""
    givens_rotations(v::AbstractVector)

For a vector `v`, return the `length(v)-1`
Givens rotations `g` and the norm `r` such that:

```julia
g * v ≈ r * [n == 1 ? 1 : 0 for n in 1:length(v)]
```
"""
function givens_rotations(v::AbstractVector{ElT}) where {ElT}
  N = length(v)
  gs = Circuit{ElT}([])
  r = v[1]
  for n in reverse(1:(N - 1))
    g, r = givens(v, n, n + 1)
    v = g * v
    lmul!(g, gs)
  end
  return gs, r
end

givens_rotations(v::ConservesNf) = return givens_rotations(v.data)

"""
  givens_rotations(_v0::ConservesNfParity)
  
  For a vector
  ```julia
  v=_v0.data
  ```
  from a fermionic Gaussian state, return the `4*length(v)-1`
  real Givens/Boguliobov rotations `g` and the norm `r` such that:
  ```julia
  g * v ≈ r * [n == 2 ? 1 : 0 for n in 1:length(v)]
 c
  with `g` being composed of diagonal rotation aligning pairs
  of complex numbers in the complex plane, and Givens/Boguliobov Rotations
  with real arguments only, acting on the interlaced single-particle space of
  annihilation and creation operator coefficients.
  """
function givens_rotations(_v0::ConservesNfParity;)
  v0 = _v0.data
  N = div(length(v0), 2)
  if N == 1
    error(
      "Givens rotation on 2-element vector not allowed for ConservesNfParity-type calculations. This should have been caught elsewhere.",
    )
  end
  ElT = eltype(v0)
  gs = Circuit{ElT}([])
  v = copy(v0)
  # detect if v is actually number-conserving because only defined in terms of annihilation operators
  if norm(v[2:2:end]) < 10 * eps(real(ElT))
    r = v[1]
    gsca, _ = givens_rotations(v[1:2:end])
    replace_indices!(i -> 2 * i - 1, gsca)
    gscc = Circuit(copy(gsca.rotations))
    replace_indices!(i -> i + 1, gsca)
    conj!(gscc)
    gsc = interleave(gscc, gsca)
    LinearAlgebra.lmul!(gsc, gs)
    return gs, r
  end
  r = v[2]
  # Given's rotations from creation-operator coefficients
  gscc, _ = givens_rotations(v[2:2:end])
  replace_indices!(i -> 2 * i, gscc)
  gsca = Circuit(copy(gscc.rotations))
  replace_indices!(i -> i - 1, gsca)
  conj!(gsca)
  gsc = interleave(gscc, gsca)
  LinearAlgebra.lmul!(gsc, gs)
  # detect if v is actually number-conserving because only defined in terms of creation operators
  if norm(v[1:2:end]) < 10 * eps(real(ElT))
    return gs, r
  end
  v = gsc * v
  # if we get here, v was actually number-non conserving, so procedure
  # Given's rotations from annihilation-operator coefficients
  gsaa, _ = givens_rotations(v[3:2:end])
  replace_indices!(i -> 2 * i + 1, gsaa)
  gsac = Circuit(copy(gsaa.rotations))
  replace_indices!(i -> i + 1, gsac)
  conj!(gsac)
  gsa = interleave(gsac, gsaa)
  v = gsa * v
  LinearAlgebra.lmul!(gsa, gs)

  # Boguliobov rotation for remaining Bell pair
  g1, r = givens(v, 2, 3)
  g2 = Givens(1, 4, g1.c, g1.s')
  v = g1 * v
  v = g2 * v #should have no effect
  LinearAlgebra.lmul!(g2, gs)
  LinearAlgebra.lmul!(g1, gs)
  return gs, r
end

function maybe_drop_pairing_correlations(Λ0::AbstractMatrix{ElT}) where {ElT<:Number}
  Λblocked = reverse_interleave(Λ0)
  N = div(size(Λblocked, 1), 2)
  if all(x -> abs(x) <= 10 * eps(real(eltype(Λ0))), @view Λblocked[1:N, (N + 1):end])
    return ConservesNf(Λblocked[(N + 1):end, (N + 1):end])
    #return ConservesNfParity(Λ0)
  else
    return ConservesNfParity(Λ0)
  end
end

maybe_drop_pairing_correlations(Λ0::ConservesNf) = Λ0
function maybe_drop_pairing_correlations(Λ0::ConservesNfParity)
  return maybe_drop_pairing_correlations(Λ0.data)
end

sortperm(x::ConservesNf) = sortperm(x.data; by=entropy)
sortperm(x::ConservesNfParity) = sortperm(x.data)

function get_error(x::ConservesNf, perm)
  n = x.data[first(perm)]
  return min(abs(n), abs(1 - n))
end
function get_error(x::ConservesNfParity, perm)
  n1 = x.data[first(perm)]
  n2 = x.data[last(perm)]
  return min(abs(n1), abs(n2))
end

function isolate_subblock_eig(
  _Λ::AbstractSymmetry,
  startind::Int;
  eigval_cutoff::Float64=1e-8,
  minblocksize::Int=2,
  maxblocksize::Int=div(size(_Λ.data, 1), 1),
)
  blocksize = 0
  err = 0.0
  p = Int[]
  ElT = eltype(_Λ.data)
  nB = eltype(_Λ.data)[]
  uB = 0.0
  ΛB = 0.0
  i = startind
  Λ = _Λ.data
  N = size(Λ, 1)
  for blocksize in minblocksize:maxblocksize
    j = min(site_stride(_Λ) * i + site_stride(_Λ) * blocksize, N)
    ΛB = @view Λ[
      (site_stride(_Λ) * i + 1 - site_stride(_Λ)):j,
      (site_stride(_Λ) * i + 1 - site_stride(_Λ)):j,
    ]

    if typeof(_Λ) <: ConservesNf
      nB, uB = eigen(Hermitian(ΛB))
    elseif typeof(_Λ) <: ConservesNfParity
      m = similar(ΛB)
      m .= ΛB
      _ΛB = maybe_drop_pairing_correlations(m)
      if typeof(_ΛB) <: ConservesNf
        nB, uB = eigen(Hermitian(_ΛB.data))
        #promote basis uB to non-conserving frame
        N2 = size(nB, 1) * 2
        nuB = zeros(eltype(uB), N2, N2)
        nuB[2:2:N2, 1:2:N2] .= uB
        nuB[1:2:N2, 2:2:N2] .= conj(uB)
        uB = nuB
        nB = interleave(1 .- nB, nB)
      elseif typeof(_ΛB) <: ConservesNfParity
        nB, uB = ITensorGaussianMPS.diag_corr_gaussian(Hermitian(ΛB))
        #try to rotate to real
        uB = ITensorGaussianMPS.make_real_if_possible(uB, nB .- 0.5)
        if ElT <: Real
          if norm(imag.(uB)) <= sqrt(eps(real(ElT)))
            uB = real(real.(uB))
          else
            error(
              "Not able to construct real fermionic basis for input correlation matrix. Exiting, retry with complex input type.",
            )
          end
        end
      end
    end
    nB = set_data(_Λ, abs.(nB))
    p = sortperm(nB)
    err = get_error(nB, p)
    err ≤ eigval_cutoff && break
  end
  v = set_data(_Λ, @view uB[:, p[1]])
  return v, nB, err
end

function set_occupations!(_ns::ConservesNf, _nB::ConservesNf, _v::ConservesNf, i::Int)
  p = Int[]
  ns = _ns.data
  nB = _nB.data
  p = sortperm(nB; by=entropy)
  ns[i] = nB[p[1]]
  return nothing
end

function set_occupations!(
  _ns::ConservesNfParity, _nB::ConservesNfParity, _v::ConservesNfParity, i::Int
)
  p = Int[]
  ns = _ns.data
  nB = _nB.data
  v = _v.data

  p = sortperm(nB)
  n1 = nB[first(p)]
  n2 = nB[last(p)]
  ns[2 * i] = n1
  ns[2 * i - 1] = n2
  if length(v) == 2
    # For some reason the last occupations are reversed, so take care of this conditionally here.
    # ToDo: Fix this in givens_rotations instead.
    if abs(v[1]) >= abs(v[2])
      ns[2 * i] = n2
      ns[2 * i - 1] = n1
    end
  end
  return nothing
end

stop_gmps_sweep(v::ConservesNfParity) = length(v.data) == 2 ? true : false
stop_gmps_sweep(v::ConservesNf) = false

"""
    correlation_matrix_to_gmps(Λ::AbstractMatrix{ElT}; eigval_cutoff::Float64 = 1e-8, maxblocksize::Int = size(Λ0, 1))

Diagonalize a correlation matrix, returning the eigenvalues and eigenvectors
stored in a structure as a set of Givens rotations.

The correlation matrix should be Hermitian, and will be treated as if it itensor
in the algorithm.

If `is_bcs`, the correlation matrix is assumed to be in interlaced format:
Λ[2*i-1:2*i,2*j-1:2*j]=[[c_i c_j^dagger , c_i c_j ], [c_i^dagger c_j^dagger,c_i^dagger c_j]]
Note that this may not be the standard choice in the literature, but it is internally
consistent with the format of single-particle Hamiltonians and Slater determinants employed.
"""

# Default to ConservesNf if no further arguments are given for backward compatibility
function correlation_matrix_to_gmps(
  Λ0::AbstractMatrix;
  eigval_cutoff::Float64=1e-8,
  minblocksize::Int=1,
  maxblocksize::Int=size(Λ0, 1),
)
  return correlation_matrix_to_gmps(
    ConservesNf(Λ0);
    eigval_cutoff=eigval_cutoff,
    minblocksize=minblocksize,
    maxblocksize=maxblocksize,
  )
end

function correlation_matrix_to_gmps(
  Λ0::AbstractMatrix,
  Nsites::Int;
  eigval_cutoff::Float64=1e-8,
  minblocksize::Int=1,
  maxblocksize::Int=size(Λ0, 1),
)
  return correlation_matrix_to_gmps(
    symmetric_correlation_matrix(Λ0, Nsites);
    eigval_cutoff=eigval_cutoff,
    minblocksize=minblocksize,
    maxblocksize=maxblocksize,
  )
end

function correlation_matrix_to_gmps(
  Λ0::T;
  eigval_cutoff::Float64=1e-8,
  minblocksize::Int=1,
  maxblocksize::Int=size(Λ0.data, 1),
) where {T<:AbstractSymmetry}
  ElT = eltype(Λ0.data)
  Λ = T(Hermitian(copy((Λ0.data))))
  V = Circuit{ElT}([])
  err_tot = 0.0 ### FIXME: keep track of error below
  N = size(Λ.data, 1)
  #ns = set_data(Λ, Vector{real(ElT)}(undef, N))
  for i in 1:div(N, site_stride(Λ))
    err = 0.0
    v, _, err = isolate_subblock_eig(
      Λ,
      i;
      eigval_cutoff=eigval_cutoff,
      minblocksize=minblocksize,
      maxblocksize=maxblocksize,
    )
    if stop_gmps_sweep(v)
      break
    end
    g, _ = givens_rotations(v)
    replace_indices!(j -> j + site_stride(Λ) * (i - 1), g)

    # In-place version of:
    # V = g * V
    LinearAlgebra.lmul!(g, V)
    Λ = set_data(Λ, Hermitian(g * Matrix(Λ.data) * g'))
  end
  ###return non-wrapped occupations for backwards compatibility
  ns = diag(Λ.data)
  @assert norm(imag.(ns)) <= sqrt(eps(real(ElT)))

  return real(real.(ns)), V
end

function (x::AbstractSymmetry * y::AbstractSymmetry)
  if !has_same_symmetry(x, y)
    error("Can't multiply two symmetric objects with different symmetries.")
  end
  return set_data(x, x.data * y.data)
end

has_same_symmetry(::AbstractSymmetry, ::AbstractSymmetry) = false
has_same_symmetry(::ConservesNf, ::ConservesNf) = true
has_same_symmetry(::ConservesNfParity, ::ConservesNfParity) = true

function slater_determinant_to_gmps(Φ::AbstractMatrix, N::Int; kwargs...)
  return correlation_matrix_to_gmps(conj(Φ) * transpose(Φ), N; kwargs...)
end

function slater_determinant_to_gmps(Φ::AbstractMatrix; kwargs...)
  return correlation_matrix_to_gmps(ConservesNf(conj(Φ) * transpose(Φ)); kwargs...)
end

function slater_determinant_to_gmps(Φ::AbstractSymmetry; kwargs...)
  return correlation_matrix_to_gmps(conj(Φ) * transpose(Φ); kwargs...)
end

#
# Turn circuit into MPS
#

function ITensors.ITensor(u::Givens, s1::Index, s2::Index)
  U = [
    1 0 0 0
    0 u.c u.s 0
    0 -conj(u.s) u.c 0
    0 0 0 1
  ]
  return itensor(U, s2', s1', dag(s2), dag(s1))
end

function ITensors.ITensor(b::Boguliobov, s1::Index, s2::Index)
  U = [
    b.u.c 0 0 conj(b.u.s)
    0 1 0 0
    0 0 1 0
    -(b.u.s) 0 0 b.u.c
  ]
  return itensor(U, s2', s1', dag(s2), dag(s1))
end

function ITensors.ITensor(sites::Vector{<:Index}, u::ConservesNfParity{Givens{T}}) where {T}
  s1 = sites[div(u.data.i1 + 1, 2)]
  s2 = sites[div(u.data.i2 + 1, 2)]
  if abs(u.data.i2 - u.data.i1) % 2 == 1
    return ITensor(Boguliobov(u.data), s1, s2)
  else
    return ITensor(u.data, s1, s2)
  end
end

function ITensors.ITensor(sites::Vector{<:Index}, u::ConservesNf{Givens{T}}) where {T}
  return ITensor(sites, u.data)
end

function ITensors.ITensor(sites::Vector{<:Index}, u::Givens)
  s1 = sites[u.i1]
  s2 = sites[u.i2]
  return ITensor(u, s1, s2)
end

function itensors(s::Vector{<:Index}, C::ConservesNfParity)
  U = [ITensor(s, set_data(C, g)) for g in reverse(C.data.rotations[begin:2:end])]
  return U
end

function itensors(sites::Vector{<:Index}, C::ConservesNf)
  return itensors(sites, C.data)
end

function itensors(s::Vector{<:Index}, C::Circuit)
  U = [ITensor(s, g) for g in reverse(C.rotations)]
  return U
end

"""
    MPS(sites::Vector{<:Index}, state, U::Vector{<:ITensor}; kwargs...)

Return an MPS with site indices `sites` by applying the circuit `U` to the starting state `state`.
"""
function ITensors.MPS(sites::Vector{<:Index}, state, U::Vector{<:ITensor}; kwargs...)
  return apply(U, productMPS(sites, state); kwargs...)
end

function isspinful(s::Index)
  !hasqns(s) && return false
  return all(qnblock -> ITensors.hasname(qn(qnblock), ITensors.QNVal("Sz", 0)), space(s))
end

function isspinful(s::Vector{<:Index})
  return all(isspinful, s)
end

# Checks whether correlation matrix is of a number conserving system and returns AbstractSymmetry wrapper around correlation matrix
# ToDo: Behaviour assumes (spinless) "Fermion" sites, handle "Electron" sites separately for cases where correlation matrix does not factorize.
function symmetric_correlation_matrix(Λ::AbstractMatrix, s::Vector{<:Index})
  if length(s) == size(Λ, 1)
    return ConservesNf(Λ)
  elseif 2 * length(s) == size(Λ, 1)
    return ConservesNfParity(Λ)
  else
    return error("Correlation matrix is not the same or twice the length of sites")
  end
end

function symmetric_correlation_matrix(Λ::AbstractMatrix, Nsites::Int)
  if Nsites == size(Λ, 1)
    return ConservesNf(Λ)
  elseif 2 * Nsites == size(Λ, 1)
    return ConservesNfParity(Λ)
  else
    return error("Correlation matrix is not the same or twice the length of sites")
  end
end

function correlation_matrix_to_mps(
  s::Vector{<:Index},
  Λ::AbstractMatrix;
  eigval_cutoff::Float64=1e-8,
  maxblocksize::Int=size(Λ, 1),
  minblocksize::Int=1,
  kwargs...,
)
  return correlation_matrix_to_mps(
    s,
    symmetric_correlation_matrix(Λ, s);
    eigval_cutoff=eigval_cutoff,
    maxblocksize=maxblocksize,
    minblocksize=minblocksize,
    kwargs...,
  )
end

"""
    correlation_matrix_to_mps(s::Vector{<:Index}, Λ::AbstractMatrix{ElT};
                              eigval_cutoff::Float64 = 1e-8,
                              maxblocksize::Int = size(Λ, 1),
                              kwargs...)

Return an approximation to the state represented by the correlation matrix as
a matrix product state (MPS).

The correlation matrix should correspond to a pure state (have all eigenvalues
of zero or one).
"""
function correlation_matrix_to_mps(
  s::Vector{<:Index},
  Λ0::AbstractSymmetry;
  eigval_cutoff::Float64=1e-8,
  maxblocksize::Int=size(Λ0.data, 1),
  minblocksize::Int=1,
  kwargs...,
)
  MPS_Elt = eltype(Λ0.data)
  Λ = maybe_drop_pairing_correlations(Λ0)
  @assert size(Λ.data, 1) == size(Λ.data, 2)
  ns, C = correlation_matrix_to_gmps(
    Λ; eigval_cutoff=eigval_cutoff, minblocksize=minblocksize, maxblocksize=maxblocksize
  )
  if all(hastags("Fermion"), s)
    U = itensors(s, set_data(Λ, C))
    ψ = MPS(MPS_Elt, s, n -> round(Int, ns[site_stride(Λ) * n]) + 1)
    ψ = apply(U, ψ; kwargs...)
  elseif all(hastags("Electron"), s)
    # ToDo: This is not tested properly, Electron sitetype tests currently assume interface with two AbstractSymmetry (correlation matrix) arguments
    # FIXME: isodd is not correct here, there shouldn't be any restrictions on the number of electronic sites.
    isodd(length(s)) && error(
      "For Electron type, must have even number of sites of alternating up and down spins.",
    )
    N = length(s)
    if isspinful(s)
      # FIXME: Can we lift this restriction now, at least for ConservesNf?
      error(
        "correlation_matrix_to_mps(Λ::AbstractMatrix) currently only supports spinless Fermions or Electrons that do not conserve Sz. Use correlation_matrix_to_mps(Λ_up::AbstractMatrix, Λ_dn::AbstractMatrix) to use spinful Fermions/Electrons.",
      )
    elseif typeof(Λ) <: ConservesNf
      sf = siteinds("Fermion", 2 * N; conserve_qns=true)
    elseif typeof(Λ) <: ConservesNfParity
      # FIXME: Does this also break, even if it doesn't make use of identity blocks? To be safe, issue error.
      error(
        "ConservesNfParity and Electron site type currently not supported. Please use Fermion sites instead.",
      )
      sf = siteinds("Fermion", 2 * N; conserve_qns=false, conserve_nfparity=true)
    end
    U = itensors(sf, set_data(Λ, C))
    ψ = MPS(MPS_Elt, sf, n -> round(Int, ns[site_stride(Λ) * n]) + 1)
    ψ = apply(U, ψ; kwargs...)
    ψ = MPS(N)
    for n in 1:N
      i, j = 2 * n - 1, 2 * n
      C = combiner(sf[i], sf[j])
      c = combinedind(C)
      ψ[n] = ψf[i] * ψf[j] * C
      ψ[n] *= δ(dag(c), s[n]) ###This back conversion to Electron will likely not work reliably for ConservesNfParity
    end
  else
    error("All sites must be Fermion or Electron type.")
  end
  return ψ
end

"""
    slater_determinant_to_mps(s::Vector{<:Index}, Φ::AbstractMatrix; kwargs...)

Given indices and matrix of orbitals representing a Slater determinant, 
compute a matrix product state (MPS) approximately having the same correlation 
matrices as this Slater determinant.

Optional keyword arguments:
* `eigval_cutoff::Float64=1E-8` - cutoff used to adaptively determine the block size (eigenvalues must be closer to 1 or 0 by an amount smaller than this cutoff for their eigenvectors be labeled as "inactive" orbitals)
* `maxblocksize::Int` - maximum block size used to compute inactive orbitals. Setting this to a smaller value can lead to faster running times and a smaller MPS bond dimension, though the accuracy may be lower.
"""
function slater_determinant_to_mps(s::Vector{<:Index}, Φ::AbstractMatrix; kwargs...)
  return correlation_matrix_to_mps(s, conj(Φ) * transpose(Φ); kwargs...)
end

function slater_determinant_to_mps(s::Vector{<:Index}, Φ::AbstractSymmetry; kwargs...)
  return correlation_matrix_to_mps(s, conj(Φ) * transpose(Φ); kwargs...)
end

function slater_determinant_to_mps(
  s::Vector{<:Index}, Φ_up::AbstractMatrix, Φ_dn::AbstractMatrix; kwargs...
)
  return correlation_matrix_to_mps(
    s, conj(Φ_up) * transpose(Φ_up), conj(Φ_dn) * transpose(Φ_dn); kwargs...
  )
end

function mapindex(f::Function, C::Circuit)
  return Circuit(mapindex.(f, C.rotations))
end

function mapindex(f::Function, g::Givens)
  return Givens(f(g.i1), f(g.i2), g.c, g.s)
end

function identity_blocks!(T::Tensor)
  # FIXME: This is not generic logic. Only works reliably for QN subspace sizes = 1.
  for b in nzblocks(T)
    T[b] = Matrix{Float64}(I, dims(T[b]))
  end
  return T
end

# Creates an ITensor with the specified flux where each nonzero block
# is identity
# TODO: make a special constructor for this.
# TODO: Introduce a modified combiner which keeps track of state-ordering/spaces.
function identity_blocks_itensor(flux::QN, i1::Index, i2::Index)
  A = ITensor(flux, i1, i2)
  identity_blocks!(tensor(A))
  return A
end

function identity_blocks_itensor(i1::ITensors.QNIndex, i2::ITensors.QNIndex)
  return identity_blocks_itensor(QN(), i1, i2)
end

function identity_blocks_itensor(i1::Index, i2::Index)
  M = Matrix{Float64}(I, dim(i1), dim(i2))
  return itensor(M, i1, i2)
end

convert_union_nothing(v::Vector{T}) where {T} = convert(Vector{Union{T,Nothing}}, v)

function interleave(xs...)
  nexts = convert_union_nothing(collect(Base.iterate.(xs)))
  res = Union{eltype.(xs)...}[]
  while any(!isnothing, nexts)
    for ii in eachindex(nexts)
      if !isnothing(nexts[ii])
        (item, state) = nexts[ii]
        push!(res, item)
        nexts[ii] = iterate(xs[ii], state)
      end
    end
  end
  return res
end

function interleave(a::ConservesNf{T}, b::ConservesNf{T}) where {T}
  return set_data(a, interleave(a.data, b.data))
end
function interleave(a::ConservesNfParity{T}, b::ConservesNfParity{T}) where {T}
  return set_data(
    a,
    interleave(
      interleave(a.data[1:2:end], b.data[1:2:end]),
      interleave(a.data[2:2:end], b.data[2:2:end]),
    ),
  )
end

function interleave(M::AbstractMatrix)
  @assert size(M, 1) == size(M, 2)
  n = div(size(M, 1), 2)
  first_half = Vector(1:n)
  second_half = Vector((n + 1):(2 * n))
  interleaved_inds = interleave(first_half, second_half)
  return M[interleaved_inds, interleaved_inds]
end

function interleave(g1::Circuit, g2::Circuit)
  return Circuit(interleave(g1.rotations, g2.rotations))
end

function reverse_interleave(M::AbstractMatrix)
  @assert size(M, 1) == size(M, 2)
  n = div(size(M, 1), 2)
  first_half = Vector(1:n)
  second_half = Vector((n + 1):(2 * n))
  interleaved_inds = interleave(first_half, second_half)
  ordered_inds = sortperm(interleaved_inds)
  return M[ordered_inds, ordered_inds]
end

function correlation_matrix_to_mps(
  s::Vector{<:Index},
  Λ_up0::AbstractSymmetry,
  Λ_dn0::AbstractSymmetry;
  eigval_cutoff::Float64=1e-8,
  maxblocksize::Int=min(size(Λ_up0, 1), size(Λ_dn0, 1)),
  kwargs...,
)
  MPS_Elt = promote_type(eltype(Λ_up0.data), eltype(Λ_dn0.data))
  Λ_up = maybe_drop_pairing_correlations(Λ_up0)
  Λ_dn = maybe_drop_pairing_correlations(Λ_dn0)
  @assert size(Λ_up.data, 1) == size(Λ_up.data, 2)
  @assert size(Λ_dn.data, 1) == size(Λ_dn.data, 2)

  if !(
    (typeof(Λ_up) <: ConservesNfParity && typeof(Λ_dn) <: ConservesNfParity) ||
    (typeof(Λ_up) <: ConservesNf && typeof(Λ_dn) <: ConservesNf)
  )
    error("Λ_up and Λ_dn have incompatible subtypes of AbstractSymmetry")
  end

  N_up = div(size(Λ_up.data, 1), site_stride(Λ_up))
  N_dn = div(size(Λ_dn.data, 1), site_stride(Λ_up))
  N = N_up + N_dn
  ns_up, C_up = correlation_matrix_to_gmps(
    Λ_up; eigval_cutoff=eigval_cutoff, maxblocksize=maxblocksize
  )
  ns_dn, C_dn = correlation_matrix_to_gmps(
    Λ_dn; eigval_cutoff=eigval_cutoff, maxblocksize=maxblocksize
  )
  C_up = mapindex(n -> 2n - 1, C_up)
  C_dn = mapindex(n -> 2n, C_dn)
  C_up_rot = set_data(Λ_up, C_up.rotations)
  C_dn_rot = set_data(Λ_dn, C_dn.rotations)
  ns_up = set_data(Λ_up, ns_up)
  ns_dn = set_data(Λ_dn, ns_dn)
  C = Circuit(interleave(C_up_rot, C_dn_rot).data)
  ns = interleave(ns_up, ns_dn).data
  if all(hastags("Fermion"), s)
    U = itensors(s, set_data(Λ_up, C))
    ψ = MPS(MPS_Elt, s, n -> round(Int, ns[site_stride(Λ_up) * n]) + 1)
    ψ = apply(U, ψ; kwargs...)
  elseif all(hastags("Electron"), s)
    @assert length(s) == N_up
    @assert length(s) == N_dn
    if isspinful(s)
      if typeof(Λ_up) <: ConservesNf
        space_up = [QN(("Nf", 0, -1), ("Sz", 0)) => 1, QN(("Nf", 1, -1), ("Sz", 1)) => 1]
        space_dn = [QN(("Nf", 0, -1), ("Sz", 0)) => 1, QN(("Nf", 1, -1), ("Sz", -1)) => 1]
      elseif typeof(Λ_up) <: ConservesNfParity
        error(
          "ConservesNfParity and Electron site type currently not supported. Please use Fermion sites instead.",
        )
        # FIXME: issue with combiner-logic for subspace-size > 1 in identity_blocks_itensor, see below
        space_up = [QN(("NfParity", 0, -2),) => 1, QN(("NfParity", 1, -2),) => 1]
        space_dn = [QN(("NfParity", 0, -2),) => 1, QN(("NfParity", 1, -2),) => 1]
      end
      sf_up = [Index(space_up, "Fermion,Site,n=$(2n-1)") for n in 1:N_up]
      sf_dn = [Index(space_dn, "Fermion,Site,n=$(2n)") for n in 1:N_dn]
      sf = collect(Iterators.flatten(zip(sf_up, sf_dn)))
    else
      if typeof(Λ_up) <: ConservesNf
        sf = siteinds("Fermion", N; conserve_qns=true, conserve_sz=false)
      elseif typeof(Λ_up) <: ConservesNfParity
        error(
          "ConservesNfParity and Electron site type currently not supported. Please use Fermion sites instead.",
        )
        sf = siteinds(
          "Fermion", N; conserve_qns=false, conserve_sz=false, conserve_nfparity=true
        )
      end
    end
    U = itensors(sf, set_data(Λ_up, C))
    ψf = MPS(MPS_Elt, sf, n -> round(Int, ns[site_stride(Λ_up) * n]) + 1)
    ψf = apply(U, ψf; kwargs...)
    ψ = MPS(N_up)
    for n in 1:N_up
      i, j = 2 * n - 1, 2 * n
      C = combiner(sf[i], sf[j])
      c = combinedind(C)
      ψ[n] = ψf[i] * ψf[j] * C
      # FIXME: combiner looses track of state ordering for QN subspaces > 1 in identity_blocks_itensor
      ψ[n] *= identity_blocks_itensor(dag(c), s[n])
    end
  else
    error("All sites must be Fermion or Electron type.")
  end

  return ψ
end

function correlation_matrix_to_mps(
  s::Vector{<:Index},
  Λ_up::AbstractMatrix,
  Λ_dn::AbstractMatrix;
  eigval_cutoff::Float64=1e-8,
  maxblocksize::Int=min(size(Λ_up, 1), size(Λ_dn, 1)),
  minblocksize::Int=1,
  kwargs...,
)
  if all(hastags("Electron"), s)
    return correlation_matrix_to_mps(
      s,
      symmetric_correlation_matrix(Λ_up, s),
      symmetric_correlation_matrix(Λ_dn, s);
      eigval_cutoff=eigval_cutoff,
      maxblocksize=maxblocksize,
      minblocksize=minblocksize,
      kwargs...,
    )
  elseif all(hastags("Fermion"), s)
    # equivalent number of electrons
    n_electrons = div(length(s), 2)
    return correlation_matrix_to_mps(
      s,
      symmetric_correlation_matrix(Λ_up, n_electrons),
      symmetric_correlation_matrix(Λ_dn, n_electrons);
      eigval_cutoff=eigval_cutoff,
      maxblocksize=maxblocksize,
      minblocksize=minblocksize,
      kwargs...,
    )
  end
end
