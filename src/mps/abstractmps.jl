abstract type AbstractMPS end

"""
    length(::MPS/MPO)

The number of sites of an MPS/MPO.
"""
Base.length(m::AbstractMPS) = length(m.data)

"""
    size::MPS/MPO)

The number of sites of an MPS/MPO, in a tuple.
"""
Base.size(m::AbstractMPS) = size(m.data)

"""
    ITensors.data(::MPS/MPO)

The Vector storage of an MPS/MPO.

This is mostly for internal usage, please let us
know if there is functionality not available for
MPS/MPO you would like.
"""
data(m::AbstractMPS) = m.data

leftlim(m::AbstractMPS) = m.llim

rightlim(m::AbstractMPS) = m.rlim

function setleftlim!(m::AbstractMPS, new_ll::Int)
  m.llim = new_ll
end

function setrightlim!(m::AbstractMPS, new_rl::Int)
  m.rlim = new_rl
end

isortho(m::AbstractMPS) = leftlim(m)+1 == rightlim(m)-1

function orthocenter(m::T) where {T<:AbstractMPS}
  !isortho(m) && error("$T has no well-defined orthogonality center")
  return leftlim(m)+1
end

Base.getindex(M::AbstractMPS,
              n::Integer) = getindex(data(M),n)

function Base.setindex!(M::AbstractMPS,
                        T::ITensor,
                        n::Integer;
                        set_limits::Bool = true)
  if set_limits
    (n <= leftlim(M)) && setleftlim!(M,n-1)
    (n >= rightlim(M)) && setrightlim!(M,n+1)
  end
  data(M)[n] = T
  return M
end

function Base.setindex!(M::MPST, v::MPST,
                        ::Colon) where {MPST <: AbstractMPS}
  setleftlim!(M, leftlim(v))
  setrightlim!(M, rightlim(v))
  data(M)[:] = data(v)
  return M
end

Base.setindex!(M::AbstractMPS, v::Vector{<:ITensor}, ::Colon) =
  setindex!(M, MPS(v), :)

Base.copy(m::AbstractMPS) = typeof(m)(copy(data(m)),
                                      leftlim(m),
                                      rightlim(m))

Base.similar(m::AbstractMPS) = typeof(m)(similar(data(m)),
                                         0,
                                         length(m))

Base.deepcopy(m::AbstractMPS) = typeof(m)(deepcopy(data(m)),
                                          leftlim(m),
                                          rightlim(m))

Base.eachindex(m::AbstractMPS) = 1:length(m)

Base.iterate(M::AbstractMPS) = iterate(data(M))

Base.iterate(M::AbstractMPS, state) = iterate(data(M), state)

"""
    unique_siteind(A::MPO, B::MPS, j::Int)
    unique_siteind(A::MPO, B::MPO, j::Int)

Get the site index of MPO `A` that is unique to `A` (not shared with MPS/MPO `B`).
"""
function unique_siteind(A::AbstractMPS, B::AbstractMPS, j::Int)
  N = length(A)
  j == 1 && return uniqueind(A[j], A[j+1], B[j])
  j == N && return uniqueind(A[j], A[j-1], B[j])
  return uniqueind(A[j], A[j-1], A[j+1], B[j])
end

"""
    unique_siteinds(A::MPO, B::MPS)
    unique_siteinds(A::MPO, B::MPO)

Get the site indices of MPO `A` that are unique to `A` (not shared with MPS/MPO `B`), as a `Vector{<:Index}`.
"""
function unique_siteinds(A::AbstractMPS, B::AbstractMPS)
  return [unique_siteind(A, B, j) for j in eachindex(A)]
end

"""
    common_siteind(A::MPO, B::MPS, j::Int)
    common_siteind(A::MPO, B::MPO, j::Int)

Get the site index of MPO `A` that is shared with MPS/MPO `B`.
"""
function common_siteind(A::AbstractMPS, B::AbstractMPS, j::Int)
  return commonind(A[j], B[j])
end

"""
    common_siteinds(A::MPO, B::MPS)
    common_siteinds(A::MPO, B::MPO)

Get the site indices of MPO `A` that are shared with MPS/MPO `B`, as a `Vector{<:Index}`.
"""
function common_siteinds(A::AbstractMPS, B::AbstractMPS)
  return [common_siteind(A, B, j) for j in eachindex(A)]
end

Base.keys(ψ::AbstractMPS) = keys(data(ψ))

#
# Find sites of an MPS or MPO
#

# TODO: accept a keyword argument sitedict that
# is a dictionary from the site indices to the site.
"""
    findsite(M::Union{MPS, MPO}, is)

Return the first site of the MPS or MPO that has at least one
Index in common with the Index or collection of indices `is`.

To find all sites with common indices with `is`, use the 
`findsites` function.

# Examples
```julia
s = siteinds("S=1/2", 5)
ψ = randomMPS(s)
findsite(ψ, s[3]) == 3
findsite(ψ, (s[3], s[4])) == 3

M = MPO(s)
findsite(M, s[4]) == 4
findsite(M, s[4]') == 4
findsite(M, (s[4]', s[4])) == 4
findsite(M, (s[4]', s[3])) == 3
```
"""
findsite(ψ::AbstractMPS, is) = findfirst(hascommoninds(is), ψ)

findsite(ψ::AbstractMPS, s::Index) = findsite(ψ, IndexSet(s))

"""
    findsites(M::Union{MPS, MPO}, is)

Return the sites of the MPS or MPO that have
indices in common with the collection of site indices
`is`.

# Examples
```julia
s = siteinds("S=1/2", 5)
ψ = randomMPS(s)
findsites(ψ, s[3]) == [3]
findsites(ψ, (s[4], s[1])) == [1, 4]

M = MPO(s)
findsites(M, s[4]) == [4]
findsites(M, s[4]') == [4]
findsites(M, (s[4]', s[4])) == [4]
findsites(M, (s[4]', s[3])) == [3, 4]
```
"""
findsites(ψ::ITensors.AbstractMPS, is) =
  findall(hascommoninds(is), ψ)

findsites(ψ::ITensors.AbstractMPS, s::Index) =
 findsites(ψ, IndexSet(s))

# TODO: depracate in favor of findsite.
"""
    findfirstsiteind(M::MPS, i::Index)

    findfirstsiteind(M::MPO, i::Index)

Return the first site of the MPS or MPO that has the
site index `i`.
"""
function findfirstsiteind(ψ::AbstractMPS,
                          s::Index)
  return findfirst(hasind(s), ψ)
end

# TODO: depracate in favor of findsite.
"""
    findfirstsiteind(M::MPS, is)

    findfirstsiteind(M::MPO, is)

Return the first site of the MPS or MPO that has the
site indices `is`.
"""
function findfirstsiteinds(ψ::AbstractMPS,
                           s)
  return findfirst(hasinds(s), ψ)
end

"""
    firstsiteind(M::Union{MPS,MPO}, j::Int; kwargs...)

Return the first site Index found on the MPS or MPO
(the first Index unique to the `j`th MPS/MPO tensor).

You can choose different filters, like prime level
and tags, with the `kwargs`.
"""
function firstsiteind(M::AbstractMPS, j::Int;
                      kwargs...)
  N = length(M)
  (N==1) && return firstind(M[1]; kwargs...)
  if j == 1
    si = uniqueind(M[j], M[j+1]; kwargs...)
  elseif j == N
    si = uniqueind(M[j], M[j-1]; kwargs...)
  else
    si = uniqueind(M[j], M[j-1], M[j+1]; kwargs...)
  end
  return si
end

"""
    siteinds(M::Union{MPS, MPO}}, j::Int; kwargs...)

Return the site Indices found of the MPO or MPO
at the site `j` as an IndexSet.

Optionally filter prime tags and prime levels with
keyword arguments like `plev` and `tags`.
"""
function siteinds(M::AbstractMPS, j::Int; kwargs...)
  N = length(M)
  (N==1) && return inds(M[1]; kwargs...)
  if j == 1
    si = uniqueinds(M[j], M[j+1]; kwargs...)
  elseif j == N
    si = uniqueinds(M[j], M[j-1]; kwargs...)
  else
    si = uniqueinds(M[j], M[j-1], M[j+1]; kwargs...)
  end
  return si
end

# TODO: change kwarg from `set_limits` to `preserve_ortho`
function Base.map!(f::Function, M::AbstractMPS;
                   set_limits::Bool = true)
  for i in eachindex(M)
    M[i, set_limits = set_limits] = f(M[i])
  end
  return M
end

# TODO: change kwarg from `set_limits` to `preserve_ortho`
Base.map(f::Function, M::AbstractMPS; set_limits::Bool = true) =
  map!(f, copy(M); set_limits = set_limits)

for fname in (:dag,
              :prime,
              :setprime,
              :noprime,
              :addtags,
              :removetags,
              :replacetags,
              :settags)
  fname_bang = Symbol(fname, :!)

  @eval begin
    """
        $($fname)(M::MPS, args...; kwargs...)

        $($fname)(M::MPO, args...; kwargs...)

    Apply $($fname) to all ITensors of an MPS/MPO, returning a new MPS/MPO.

    The ITensors of the MPS/MPO will be a view of the storage of the original ITensors.
    """
    $fname(M::AbstractMPS, args...;
           set_limits::Bool = false, kwargs...) =
      map(m -> $fname(m, args...; kwargs...), M;
          set_limits = set_limits)

    """
        $($fname_bang)(M::MPS, args...; kwargs...)

        $($fname_bang)(M::MPO, args...; kwargs...)

    Apply $($fname) to all ITensors of an MPS/MPO in-place.
    """
    $fname_bang(M::AbstractMPS, args...;
                set_limits::Bool = false, kwargs...) =
      map!(m -> $fname(m, args...; kwargs...), M;
           set_limits = set_limits)
  end
end

function map_linkinds!(f::Function, M::AbstractMPS)
  for i in eachindex(M)[1:end-1]
    l = linkind(M, i)
    if !isnothing(l)
      l̃ = f(l)
      M[i, set_limits = false] = replaceind(M[i], l, l̃)
      M[i+1, set_limits = false] = replaceind(M[i+1], l, l̃)
    end
  end
  return M
end

map_linkinds(f::Function, M::AbstractMPS) = map_linkinds!(f, copy(M))

function map_common_siteinds!(f::Function, M1::AbstractMPS,
                                           M2::AbstractMPS)
  length(M1) != length(M2) && error("MPOs/MPSs must be the same length")
  for i in eachindex(M1)
    s = common_siteind(M1, M2, i)
    if !isnothing(s)
      s̃ = f(s)
      M1[i, set_limits = false] = replaceind(M1[i], s, s̃)
      M2[i, set_limits = false] = replaceind(M2[i], s, s̃)
    end
  end
  return M1, M2
end

function map_common_siteinds(f::Function, M1::AbstractMPS,
                                          M2::AbstractMPS)
  return map_common_siteinds!(f, copy(M1), copy(M2))
end

function map_unique_siteinds!(f::Function, M1::AbstractMPS,
                                           M2::AbstractMPS)
  length(M1) != length(M2) && error("MPOs/MPSs must be the same length")
  for i in eachindex(M1)
    s = unique_siteind(M1, M2, i)
    if !isnothing(s)
      s̃ = f(s)
      M1[i, set_limits = false] = replaceind(M1[i], s, s̃)
    end
  end
  return M1
end

function map_unique_siteinds(f::Function, M1::AbstractMPS,
                                        M2::AbstractMPS)
  return map_unique_siteinds!(f, copy(M1), M2)
end

for fname in (:sim,
              :prime,
              :setprime,
              :noprime,
              :addtags,
              :removetags,
              :replacetags,
              :settags)
  fname_linkinds = Symbol(fname, :_linkinds)
  fname_linkinds_inplace = Symbol(fname_linkinds, :!)
  fname_common_siteinds = Symbol(fname, :_common_siteinds)
  fname_common_siteinds_inplace = Symbol(fname_common_siteinds, :!)
  fname_unique_siteinds = Symbol(fname, :_unique_siteinds)
  fname_unique_siteinds_inplace = Symbol(fname_unique_siteinds, :!)

  @eval begin
    """
        $($fname_linkinds)(M::MPS, args...; kwargs...)

        $($fname_linkinds)(M::MPO, args...; kwargs...)

    Apply $($fname) to all link indices of an MPS/MPO, returning a new MPS/MPO.
    
    The ITensors of the MPS/MPO will be a view of the storage of the original ITensors.
    """
    $fname_linkinds(M::AbstractMPS, args...; kwargs...) =
      map_linkinds(i -> $fname(i, args...; kwargs...), M)

    """
        $($fname_linkinds)!(M::MPS, args...; kwargs...)

        $($fname_linkinds)!(M::MPO, args...; kwargs...)

    Apply $($fname) to all link indices of the ITensors of an MPS/MPO in-place.
    """
    $fname_linkinds_inplace(M::AbstractMPS, args...; kwargs...) =
      map_linkinds!(i -> $fname(i, args...; kwargs...), M)

    """
        $($fname_common_siteinds)(M1::MPO, M2::MPS, args...; kwargs...)

        $($fname_common_siteinds)(M1::MPO, M2::MPO, args...; kwargs...)

    Apply $($fname) to the site indices that are shared by `M1` and `M2`.
    
    Returns new MPSs/MPOs. The ITensors of the MPSs/MPOs will be a view of the storage of the original ITensors.
    """
    function $fname_common_siteinds(M1::AbstractMPS,
                                    M2::AbstractMPS,
                                    args...;
                                    kwargs...)
      return map_common_siteinds(i -> $fname(i, args...;
                                             kwargs...), M1, M2)
    end

    """
        $($fname_common_siteinds)!(M1::MPO, M2::MPS, args...; kwargs...)

        $($fname_common_siteinds)!(M1::MPO, M2::MPO, args...; kwargs...)

    Apply $($fname) to the site indices that are shared by `M1` and `M2`. Returns new MPSs/MPOs.
    
    Modifies the input MPSs/MPOs in-place.
    """
    $fname_common_siteinds_inplace(M1::AbstractMPS, M2::AbstractMPS,
                                   args...; kwargs...) =
      map_common_siteinds!(i -> $fname(i, args...; kwargs...), M1, M2)

    """
        $($fname_unique_siteinds)(M1::MPO, M2::MPS, args...; kwargs...)

    Apply $($fname) to the site indices of `M1` that are not shared with `M2`. Returns new MPSs/MPOs.
    
    The ITensors of the MPSs/MPOs will be a view of the storage of the original ITensors.
    """
    function $fname_unique_siteinds(M1::AbstractMPS,
                                    M2::AbstractMPS,
                                    args...;
                                    kwargs...)
      return map_unique_siteinds(i -> $fname(i, args...;
                                             kwargs...), M1, M2)
    end

    """
        $($fname_unique_siteinds)!(M1::MPO, M2::MPS, args...; kwargs...)

    Apply $($fname) to the site indices of `M1` that are not shared with `M2`. Modifies the input MPSs/MPOs in-place.
    """
    function $fname_unique_siteinds_inplace(M1::AbstractMPS,
                                            M2::AbstractMPS,
                                            args...;
                                            kwargs...)
      return map_unique_siteinds!(i -> $fname(i, args...;
                                              kwargs...), M1, M2)
    end
  end
end


"""
    maxlinkdim(M::MPS)

    maxlinkdim(M::MPO)

Get the maximum link dimension of the MPS or MPO.
"""
function maxlinkdim(M::AbstractMPS)
  md = 0
  for b ∈ eachindex(M)[1:end-1]
    l = linkind(M, b)
    linkdim = isnothing(l) ? 0 : dim(l)
    md = max(md, linkdim)
  end
  md
end

"""
    linkind(M::MPS, j::Int)

    linkind(M::MPO, j::Int)

Get the link or bond Index connecting the
MPS or MPO tensor on site j to site j+1.

If there is no link Index, return `nothing`.
"""
function linkind(M::AbstractMPS, j::Int)
  N = length(M)
  (j ≥ length(M) || j < 1) && return nothing
  return commonind(M[j], M[j+1])
end

linkinds(ψ::AbstractMPS) =
  [linkind(ψ, b) for b in 1:length(ψ)-1]

"""
    linkdim(M::MPS, j::Int)

    linkdim(M::MPO, j::Int)

Get the dimension of the link or bond connecting the
MPS or MPO tensor on site j to site j+1.

If there is no link Index, return `nothing`.
"""
function linkdim(ψ::AbstractMPS, b::Int)
  l = linkind(ψ, b)
  isnothing(l) && return nothing
  return dim(l)
end

linkdims(ψ::AbstractMPS) =
  [linkdim(ψ, b) for b in 1:length(ψ)-1]

function _log_or_not_dot(M1::MPST,
                         M2::MPST,
                         loginner::Bool;
                         make_inds_match::Bool = true)::Number where {MPST <: AbstractMPS}
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

  if loginner
    normO = norm(O)
    log_inner_tot = log(normO)
    O ./= normO
  end

  for j in eachindex(M1)[2:end]
    O = (O*M1dag[j])*M2[j]

    if loginner
      normO = norm(O)
      log_inner_tot += log(normO)
      O ./= normO
    end

  end

  if loginner
    return log_inner_tot
  end

  return O[]
end

"""
    dot(A::MPS, B::MPS; make_inds_match = true)
    inner(A::MPS, B::MPS; make_inds_match = true)

    dot(A::MPO, B::MPO)
    inner(A::MPO, B::MPO)

Compute the inner product `<A|B>`. If `A` and `B` are MPOs, computes the Frobenius inner product.

If `make_inds_match = true`, the function attempts to make
the site indices match before contracting (so for example, the
inputs can have different site indices, as long as they
have the same dimensions or QN blocks).

For now, `make_inds_match` is only supported for MPSs.

See also `logdot`/`loginner`.
"""
function LinearAlgebra.dot(M1::MPST,
                           M2::MPST;
                           kwargs...) where {MPST <: AbstractMPS}
  return _log_or_not_dot(M1, M2, false; kwargs...)
end

"""
    logdot(A::MPS, B::MPS; make_inds_match = true)
    loginner(A::MPS, B::MPS; make_inds_match = true)

    logdot(A::MPO, B::MPO)
    loginner(A::MPO, B::MPO)

Compute the logarithm of the inner product `<A|B>`. If `A` and `B` are MPOs, computes the logarithm of the Frobenius inner product.

This is useful for larger MPS/MPO, where in the limit of large numbers of sites the inner product can diverge or approach zero.

If `make_inds_match = true`, the function attempts to make
the site indices match before contracting (so for example, the
inputs can have different site indices, as long as they
have the same dimensions or QN blocks).

For now, `make_inds_match` is only supported for MPSs.
"""
function logdot(M1::MPST,
                M2::MPST;
                kwargs...) where {MPST <: AbstractMPS}
  return _log_or_not_dot(M1, M2, true; kwargs...)
end

inner(M1::MPST,
      M2::MPST;
      kwargs...) where {MPST <: AbstractMPS} = dot(M1, M2; kwargs...)

loginner(M1::MPST,
         M2::MPST;
         kwargs...) where {MPST <: AbstractMPS} = logdot(M1, M2; kwargs...)

"""
    norm(A::MPS)

    norm(A::MPO)

Compute the norm of the MPS or MPO.

See also `lognorm`.
"""
function LinearAlgebra.norm(M::AbstractMPS)
  return sqrt(dot(M, M))
end

"""
    lognorm(A::MPS)

    lognorm(A::MPO)

Compute the logarithm of the norm of the MPS or MPO. 

This is useful for larger MPS/MPO that are not gauged, where in the limit of large numbers of sites the norm can diverge or approach zero.

See also `norm` and `loginner`/`logdot`.
"""
function lognorm(M::AbstractMPS)
  return 0.5 * logdot(M, M)
end

function plussers(::Type{T},
                  left_ind::Index,
                  right_ind::Index,
                  sum_ind::Index) where {T<:Array}
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
end

"""
    add(A::MPS, B::MPS; kwargs...)
    +(A::MPS, B::MPS; kwargs...)

    add(A::MPO, B::MPO; kwargs...)
    +(A::MPO, B::MPO; kwargs...)

Add two MPS/MPO with each other, with some optional
truncation.
"""
function Base.:+(A::T, B::T; kwargs...) where {T <: AbstractMPS}
  A = copy(A)
  B = copy(B)
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
    f, s = plussers(typeof(data(A[1])), lA, lB, r)
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

add(A::T, B::T;
    kwargs...) where {T <: AbstractMPS} = +(A, B; kwargs...)

"""
    sum(A::Vector{MPS}; kwargs...)

    sum(A::Vector{MPO}; kwargs...)

Add multiple MPS/MPO with each other, with some optional
truncation.
"""
function Base.sum(A::Vector{T};
                  kwargs...) where {T <: AbstractMPS}
  length(A) == 0 && return T()
  length(A) == 1 && return A[1]
  length(A) == 2 && return +(A[1], A[2]; kwargs...)
  nsize = isodd(length(A)) ? (div(length(A) - 1, 2) + 1) : div(length(A), 2)
  newterms = Vector{T}(undef, nsize)
  np = 1
  for n in 1:2:length(A) - 1
    newterms[np] = +(A[n], A[n+1]; kwargs...)
    np += 1
  end
  if isodd(length(A))
    newterms[nsize] = A[end]
  end
  return sum(newterms; kwargs...)
end

"""
    orthogonalize!(M::MPS, j::Int; kwargs...)
    orthogonalize(M::MPS, j::Int; kwargs...)

    orthogonalize!(M::MPO, j::Int; kwargs...)
    orthogonalize(M::MPO, j::Int; kwargs...)

Move the orthogonality center of the MPS
to site `j`. No observable property of the
MPS will be changed, and no truncation of the
bond indices is performed. Afterward, tensors
`1,2,...,j-1` will be left-orthogonal and tensors
`j+1,j+2,...,N` will be right-orthogonal.

Either modify in-place with `orthogonalize!` or
out-of-place with `orthogonalize`.
"""
function orthogonalize!(M::AbstractMPS,
                        j::Int;
                        kwargs...)
  while leftlim(M) < (j-1)
    (leftlim(M) < 0) && setleftlim!(M, 0)
    b = leftlim(M)+1
    linds = uniqueinds(M[b], M[b+1])
    ltags = tags(linkind(M, b))
    L,R = factorize(M[b], linds; tags = ltags, kwargs...)
    M[b] = L
    M[b+1] *= R

    setleftlim!(M,b)
    if rightlim(M) < leftlim(M)+2
      setrightlim!(M, leftlim(M)+2)
    end
  end

  N = length(M)

  while rightlim(M) > (j+1)
    (rightlim(M) > (N+1)) && setrightlim!(M,N+1)
    b = rightlim(M)-2
    rinds = uniqueinds(M[b+1],M[b])
    ltags = tags(linkind(M, b))
    L,R = factorize(M[b+1], rinds; tags = ltags, kwargs...)
    M[b+1] = L
    M[b] *= R

    setrightlim!(M,b+1)
    if leftlim(M) > rightlim(M)-2
      setleftlim!(M, rightlim(M)-2)
    end
  end
  return M
end

function orthogonalize(ψ0::AbstractMPS, args...; kwargs...)
  ψ = copy(ψ0)
  orthogonalize!(ψ, args...; kwargs...)
  return ψ
end

"""
    truncate!(M::MPS; kwargs...)

    truncate!(M::MPO; kwargs...)

Perform a truncation of all bonds of an MPS/MPO,
using the truncation parameters (cutoff,maxdim, etc.)
provided as keyword arguments.
"""
function NDTensors.truncate!(M::AbstractMPS;
                             kwargs...)
  N = length(M)

  # Left-orthogonalize all tensors to make
  # truncations controlled
  orthogonalize!(M, N)

  # Perform truncations in a right-to-left sweep
  for j in reverse(2:N)
    rinds = uniqueinds(M[j], M[j-1])
    ltags = tags(commonind(M[j], M[j-1]))
    U, S, V = svd(M[j], rinds; lefttags = ltags, kwargs...)
    M[j] = U
    M[j-1] *= (S * V)
    setrightlim!(M, j)
  end
  return M
end

NDTensors.contract(A::AbstractMPS,
                   B::AbstractMPS;
                   kwargs...) = *(A, B; kwargs...)

"""
    *(x::Number, M::MPS)

    *(x::Number, M::MPO)

Scale the MPS or MPO by the provided number.

Note: right now this just naively scales the
middle tensor. In the future, the plan would be
to scale the tensors between the left limit
and right limit by `x^(1/N)` where `N` is the distance 
between the left limit and right limit.
"""
function Base.:*(x::Number, M::AbstractMPS)
  N = deepcopy(M)
  c = div(length(N), 2)
  N[c] .*= x
  return N
end

Base.:-(M::AbstractMPS) = Base.:*(-1,M)

"""
    setindex!(::Union{MPS, MPO}, ::Union{MPS, MPO},
              r::UnitRange{Int64})

Sets a contiguous range of MPS/MPO tensors
"""
function Base.setindex!(ψ::MPST, ϕ::MPST,
                        r::UnitRange{Int64}) where {MPST <: AbstractMPS}
  @assert length(r) == length(ϕ)
  # TODO: accept r::Union{AbstractRange{Int}, Vector{Int}}
  # if r isa AbstractRange
  #   @assert step(r) = 1
  # else
  #   all(==(1), diff(r))
  # end
  llim = leftlim(ψ)
  rlim = rightlim(ψ)
  for (j, n) in enumerate(r)
    ψ[n] = ϕ[j]
  end
  if llim + 1 ≥ r[1]
    setleftlim!(ψ, leftlim(ϕ) + r[1] - 1)
  end
  if rlim - 1 ≤ r[end]
    setrightlim!(ψ, rightlim(ϕ) + r[1] - 1)
  end
  return ψ
end

_isodd_fermionic_parity(s::Index, ::Int) = false

function _isodd_fermionic_parity(s::QNIndex, n::Int)
  qn_n = qn(space(s)[n])
  fermionic_qn_pos = findfirst(q -> isfermionic(q), qn_n)
  isnothing(fermionic_qn_pos) && return false
  return isodd(val(qn_n[fermionic_qn_pos]))
end

# TODO: this version is incorrect, since it requires
# putting minus signs on the subspace of `s` that
# has 2 Fermions (not odd fermions).
# function _fermionic_swap(s::Index)
#   T = diagITensor(1, s', dag(s))
#   for b in nzblocks(T)
#     n = b[2]
#     if _isodd_fermionic_parity(s, n)
#       NDTensors.data(blockview(tensor(T), b)) .= -1
#     end
#   end
#   return T
# end

function _fermionic_swap(s1::Index, s2::Index)
  T = ITensor(s1', s2', dag(s1), dag(s2))
  dval = 1.0
  for b in nzblocks(T)
    # Must be a diagonal block
    ((b[1] ≠ b[3]) || (b[2] ≠ b[4])) && continue
    n1, n2 = b[1], b[2]
    if _isodd_fermionic_parity(s1, n1) && _isodd_fermionic_parity(s2, n2)
      dval = -1.0
    end
    Tb = blockview(tensor(T), b)
    mat_dim = prod(dims(Tb)[1:2])
    Tbr = reshape(Tb, mat_dim, mat_dim)
    for i in diagind(Tbr)
      NDTensors.setdiagindex!(Tbr, dval, i)
    end
  end
  return T
end

# TODO: add a version that determines the sites
# from common site indices of ψ and A
"""
    setindex!(ψ::Union{MPS, MPO},
              A::ITensor,
              r::UnitRange{Int};
              orthocenter::Int = last(r),
              perm = nothing,
              kwargs...)

    replacesites!([...])

    replacesites([...])

Replace the sites in the range `r` with tensors made
from decomposing `A` into an MPS or MPO.

The MPS or MPO must be orthogonalized such that
```
firstsite ≤ ITensors.orthocenter(ψ) ≤ lastsite
```

Choose the new orthogonality center with `orthocenter`, which
should be within `r`.

Optionally, permute the order of the sites with `perm`.
"""
function Base.setindex!(ψ::MPST,
                        A::ITensor,
                        r::UnitRange{Int};
                        orthocenter::Int = last(r),
                        perm = nothing,
                        kwargs...) where {MPST <: AbstractMPS}
  # Replace the sites of ITensor ψ
  # with the tensor A, splitting up A
  # into MPS tensors
  firstsite = first(r)
  lastsite = last(r)
  @assert firstsite ≤ ITensors.orthocenter(ψ) ≤ lastsite
  @assert firstsite ≤ leftlim(ψ) + 1
  @assert rightlim(ψ) - 1 ≤ lastsite

  # TODO: allow orthocenter outside of this
  # range, and orthogonalize/truncate as needed
  @assert firstsite ≤ orthocenter ≤ lastsite

  # Check that A has the proper common
  # indices with ψ
  lind = linkind(ψ, firstsite-1)
  rind = linkind(ψ, lastsite)

  sites = [siteinds(ψ, j) for j in firstsite:lastsite]

  #s = collect(Iterators.flatten(sites))
  indsA = filter(x -> !isnothing(x),
                 [lind, Iterators.flatten(sites)..., rind])
  @assert hassameinds(A, indsA)

  # For MPO case, restrict to 0 prime level
  #sites = filter(hasplev(0), sites)

  if !isnothing(perm)
    sites0 = sites
    sites = sites0[[perm...]]
    # Check if the site indices
    # are fermionic
    if any(anyfermionic, sites)
      if length(sites) == 2 && ψ isa MPS
        if all(allfermionic, sites)
          s0 = Index.(sites0)

          # TODO: the Fermionic swap is could be diagonal,
          # if we combine the site indices
          #C = combiner(s0[1], s0[2])
          #c = combinedind(C)
          #AC = A * C
          #AC = noprime(AC * _fermionic_swap(c))
          #A = AC * dag(C)

          FSWAP = _fermionic_swap(s0[1], s0[2])
          A = noprime(A * FSWAP)
        end
      elseif ψ isa MPO
        @warn "In setindex!(MPO, ::ITensor, ::UnitRange), " *
              "fermionic signs are only not handled properly for non-trivial " *
              "permutations of sites. Please inform the developers of ITensors " *
              "if you require this feature (otherwise, fermionic signs can be " *
              "put in manually with fermionic swap gates)."
      else
        @warn "In setindex!(::Union{MPS, MPO}, ::ITensor, ::UnitRange), " *
              "fermionic signs are only handled properly for permutations involving 2 sites. " *
              "The original sites are $sites0, with a permutation $perm. " *
              "To have the fermion sign handled correctly, we recommend performing your permutation " *
              "pairwise."
      end
    end
  end

  ψA = MPST(A, sites;
            leftinds = lind,
            orthocenter = orthocenter - first(r) + 1,
            kwargs...)
  #@assert prod(ψA) ≈ A

  ψ[firstsite:lastsite] = ψA

  return ψ
end

Base.setindex!(ψ::MPST,
               A::ITensor,
               r::UnitRange{Int},
               args::Pair{Symbol}...;
               kwargs...) where {MPST <: AbstractMPS} =
  setindex!(ψ, A, r; args..., kwargs...)

replacesites!(ψ::AbstractMPS, args...; kwargs...) =
  setindex!(ψ, args...; kwargs...)

replacesites(ψ::AbstractMPS, args...; kwargs...) =
  setindex!(copy(ψ), args...; kwargs...)

_number_inds(s::Index) = 1
_number_inds(s::IndexSet) = length(s)
_number_inds(sites) = sum(_number_inds(s) for s in sites)

"""
    MPS(A::ITensor, sites; <keyword arguments>)

    MPO(A::ITensor, sites; <keyword arguments>)

Construct an MPS/MPO from an ITensor `A` by decomposing it site
by site according to the site indices `sites`.

# Arguments
- `leftinds = nothing`: optional left dangling indices. Indices that are not in `sites` and `leftinds` will be dangling off of the right side of the MPS/MPO.
- `orthocenter::Int = length(sites)`: the desired final orthogonality center of the output MPS/MPO.
- `cutoff`: the desired truncation error at each link.
- `maxdim`: the maximum link dimension.
"""
function (::Type{MPST})(A::ITensor, sites;
                        leftinds = nothing,
                        orthocenter::Int = length(sites),
                        kwargs...) where {MPST <: AbstractMPS}
  N = length(sites)
  for s in sites
    @assert hasinds(A, s)
  end
  @assert isnothing(leftinds) || hasinds(A, leftinds)

  @assert 1 ≤ orthocenter ≤ N

  ψ = Vector{ITensor}(undef, N)
  Ã = A
  l = leftinds
  # TODO: To minimize work, loop from
  # 1:orthocenter and reverse(orthocenter:N)
  # so the orthogonality center is set correctly.
  for n in 1:N-1
    Lis = IndexSet(sites[n])
    if !isnothing(l)
      Lis = unioninds(Lis, l)
    end
    L, R = factorize(Ã, Lis; kwargs..., ortho = "left")
    l = commonind(L, R)
    ψ[n] = L
    Ã = R
  end
  ψ[N] = Ã
  M = MPST(ψ)
  setleftlim!(M, N-1)
  setrightlim!(M, N+1)
  orthogonalize!(M, orthocenter)
  return M
end

"""
    swapbondsites(ψ::Union{MPS, MPO}, b::Int; kwargs...)

Swap the sites `b` and `b+1`.
"""
function swapbondsites(ψ::AbstractMPS, b::Int; kwargs...)
  ortho = get(kwargs, :ortho, "right")
  ψ = copy(ψ)
  if ortho == "left"
    orthocenter = b + 1
  elseif ortho == "right"
    orthocenter = b
  end
  if leftlim(ψ) < b - 1
    orthogonalize!(ψ, b)
  elseif rightlim(ψ) > b + 2
    orthogonalize!(ψ, b + 1)
  end
  ψ[b:b + 1,
    orthocenter = orthocenter,
    perm = [2, 1], kwargs...] = ψ[b] * ψ[b + 1]
  return ψ
end

"""
    movesite(::Union{MPS, MPO}, n1n2::Pair{Int, Int})

Create a new MPS/MPO where the site at `n1` is moved to `n2`,
for a pair `n1n2 = n1 => n2`.

This is done with a series a pairwise swaps, and can introduce
a lot of entanglement into your state, so use with caution.
"""
function movesite(ψ::AbstractMPS, n1n2::Pair{Int, Int};
                  orthocenter::Int = last(n1n2),
                  kwargs...)
  n1, n2 = n1n2
  n1 == n2 && return copy(ψ)
  ψ = orthogonalize(ψ, n2)
  r = n1:n2-1
  ortho = "left"
  if n1 > n2
    r = reverse(n2:n1-1)
    ortho = "right"
  end
  for n in r
    ψ = swapbondsites(ψ, n; ortho = ortho, kwargs...)
  end
  ψ = orthogonalize(ψ, orthocenter)
  return ψ
end

# Helper function for permuting a vector for the 
# movesites function.
function _movesite(ns::Vector{Int},
                   n1n2::Pair{Int, Int})
  n1, n2 = n1n2
  n1 == n2 && return copy(ns)
  r = n1:n2-1
  if n1 > n2
    r = reverse(n2:n1-1)
  end
  for n in r
    ns = replace(ns, n => n+1, n+1 => n)
  end
  return ns
end

function _movesites(ψ::AbstractMPS, ns::Vector{Int}, ns′::Vector{Int};
                    kwargs...)
  ψ = copy(ψ)
  N = length(ns)
  @assert N == length(ns′)
  for i in 1:N
    ψ = movesite(ψ, ns[i] => ns′[i]; kwargs...)
    ns = _movesite(ns, ns[i] => ns′[i])
  end
  return ψ, ns
end

# TODO: make a permutesites(::MPS/MPO, perm)
# function that takes a permutation of the sites
# p(1:N) for N sites
function movesites(ψ::AbstractMPS,
                   nsns′::Vector{Pair{Int, Int}}; kwargs...)
  ns = first.(nsns′)
  ns′ = last.(nsns′)
  ψ = copy(ψ)
  N = length(ns)
  @assert N == length(ns′)
  p = sortperm(ns′)
  ns = ns[p]
  ns′ = ns′[p]
  ns = collect(ns)
  while ns ≠ ns′
    ψ, ns = _movesites(ψ, ns, ns′; kwargs...)
  end
  return ψ
end

# TODO: call the Vector{Pair{Int, Int}} version
function movesites(ψ::AbstractMPS,
                   ns, ns′; kwargs...)
  ψ = copy(ψ)
  N = length(ns)
  @assert N == length(ns′)
  p = sortperm(ns′)
  ns = ns[p]
  ns′ = ns′[p]
  ns = collect(ns)
  for i in 1:N
    ψ = movesite(ψ, ns[i] => ns′[i]; kwargs...)
    ns = _movesite(ns, ns[i] => ns′[i])
  end
  return ψ
end

"""
    product(o::ITensor, ψ::Union{MPS, MPO}, [ns::Vector{Int}]; <keyword argument>)

Get the product of the operator `o` with the MPS/MPO `ψ`,
where the operator is applied to the sites `ns`. If `ns`
are not specified, the sites are determined by the common indices
between `o` and the site indices of `ψ`.

If `ns` are non-contiguous, the sites of the MPS are
moved to be contiguous. By default, the sites are moved
back to their original locations. You can leave them where
they are by setting the keyword argument `move_sites_back`
to false.

# Arguments
- `move_sites_back::Bool = true`: after the ITensor is applied to the MPS or MPO, move the sites of the MPS or MPO back to their original locations.
"""
function product(o::ITensor,
                 ψ::AbstractMPS,
                 ns = findsites(ψ, o);
                 move_sites_back::Bool = true,
                 apply_dag::Bool = false,
                 kwargs...)
  N = length(ns)
  ns = sort(ns)

  # TODO: make this smarter by minimizing
  # distance to orthogonalization.
  # For example, if ITensors.orthocenter(ψ) > ns[end],
  # set to ns[end].
  ψ = orthogonalize(ψ, ns[1])
  diff_ns = diff(ns)
  ns′ = ns
  if any(!=(1), diff_ns)
    ns′ = [ns[1] + n - 1 for n in 1:N]
    ψ = movesites(ψ, ns .=> ns′; kwargs...)
  end
  ϕ = ψ[ns′[1]]
  for n in 2:N
    ϕ *= ψ[ns′[n]]
  end
  ϕ = product(o, ϕ; apply_dag = apply_dag)
  ψ[ns′[1]:ns′[end], kwargs...] = ϕ
  if move_sites_back
    # Move the sites back to their original positions
    ψ = movesites(ψ, ns′ .=> ns; kwargs...)
  end
  return ψ
end

"""
    product(As::ITensor..., M::Union{MPS, MPO})

    product(As::Vector{<:ITensor}, M::Union{MPS, MPO})

Product the ITensors `As` with the MPS or MPO `M`.

The order of operations are right associative, so for example:
`product(A1, A2, ψ) == product(A1, product(A2, ψ))`.
"""
function product(As::Vector{ <: ITensor}, ψ::AbstractMPS;
                 move_sites_back::Bool = true, kwargs...)
  Aψ = ψ
  for A in Iterators.reverse(As)
    Aψ = product(A, Aψ; move_sites_back = false, kwargs...)
  end
  if move_sites_back
    s = siteinds(Aψ)
    ns = 1:length(ψ)
    ñs = [findsite(ψ, i) for i in s]
    Aψ = movesites(Aψ, ns .=> ñs; kwargs...)
  end
  return Aψ
end

function product(Asψ::Union{ITensor, AbstractMPS}...; kwargs...)
  ψ = Asψ[end]
  @assert ψ isa AbstractMPS
  As = collect(Asψ[1:end-1])
  return product(As, ψ; kwargs...)
end

"""
    hasqns(M::MPS)

    hasqns(M::MPO)

Return true if the MPS or MPO has
tensors which carry quantum numbers.
"""
hasqns(M::AbstractMPS) = hasqns(M[1])

"""
    flux(M::MPS)

    flux(M::MPO)

    totalqn(M::MPS)

    totalqn(M::MPO)

For an MPS or MPO which conserves quantum
numbers, compute the total QN flux. For
a tensor network such as an MPS or MPO,
the flux is the sum of fluxes of each of
the tensors in the network. The name
`totalqn` is an alias for `flux`.
"""
function flux(M::AbstractMPS)
  hasqns(M) || return nothing
  q = QN()
  for j=M.llim+1:M.rlim-1
    q += flux(M[j])
  end
  return q
end

totalqn(M::AbstractMPS) = flux(M)

function checkflux(M::AbstractMPS)
  for m in M
    checkflux(m)
  end
  return nothing
end

function Base.show(io::IO, M::AbstractMPS)
  print(io,"$(typeof(M))")
  (length(M) > 0) && print(io,"\n")
  for (i, A) ∈ enumerate(data(M))
    if order(A) != 0
      println(io,"[$i] $(inds(A))")
    else
      println(io,"[$i] ITensor()")
    end
  end
end

@deprecate orthoCenter(args...;
                       kwargs...) orthocenter(args...; kwargs...)

import .NDTensors.store

@deprecate store(m::AbstractMPS) data(m)

@deprecate replacesites!(args...;
                         kwargs...) ITensors.replace_siteinds!(args...; kwargs...)

@deprecate applyMPO(args...; kwargs...) contract(args...; kwargs...)

@deprecate applympo(args...; kwargs...) contract(args...; kwargs...)

@deprecate errorMPOprod(args...;
                        kwargs...) error_contract(args...;
                                                  kwargs...)

@deprecate error_mpoprod(args...;
                         kwargs...) error_contract(args...;
                                                   kwargs...)

@deprecate error_mul(args...;
                     kwargs...) error_contract(args...;
                                               kwargs...)

@deprecate multMPO(args...; kwargs...) contract(args...; kwargs...)

import Base.sum

@deprecate sum(A::AbstractMPS,
               B::AbstractMPS; kwargs...) add(A, B; kwargs...)

@deprecate multmpo(args...;
                   kwargs...) contract(args...; kwargs...)

@deprecate set_leftlim!(args...;
                        kwargs...) ITensors.setleftlim!(args...;
                                                        kwargs...)

@deprecate set_rightlim!(args...;
                         kwargs...) ITensors.setrightlim!(args...;
                                                          kwargs...)

@deprecate tensors(args...;
                   kwargs...) ITensors.data(args...; kwargs...)

@deprecate primelinks!(args...;
                       kwargs...) ITensors.prime_linkinds!(args...;
                                                          kwargs...)

@deprecate simlinks!(args...;
                     kwargs...) ITensors.sim_linkinds!(args...;
                                                      kwargs...)

@deprecate mul(A::AbstractMPS,
               B::AbstractMPS;
               kwargs...) contract(A, B; kwargs...)

