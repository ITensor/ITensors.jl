abstract type AbstractMPS end

"""
    length(::MPS/MPO)

The number of sites of an MPS/MPO.
"""
length(m::AbstractMPS) = length(data(m))

"""
    size(::MPS/MPO)

The number of sites of an MPS/MPO, in a tuple.
"""
size(m::AbstractMPS) = size(data(m))

ndims(m::AbstractMPS) = ndims(data(m))

"""
    ITensors.data(::MPS/MPO)

Returns a view of the Vector storage of an MPS/MPO.

This is not exported and mostly for internal usage, please let us
know if there is functionality not available for MPS/MPO you would like.
"""
data(m::AbstractMPS) = m.data

leftlim(m::AbstractMPS) = m.llim

rightlim(m::AbstractMPS) = m.rlim

function setleftlim!(m::AbstractMPS, new_ll::Integer)
  m.llim = new_ll
end

function setrightlim!(m::AbstractMPS, new_rl::Integer)
  m.rlim = new_rl
end

"""
    ortho_lims(::MPS/MPO)

Returns the range of sites of the orthogonality center of the MPS/MPO.

# Examples

```julia
s = siteinds("S=½", 5)
ψ = randomMPS(s)
orthogonalize!(ψ, 3)

# ortho_lims(ψ) = 3:3
@show ortho_lims(ψ)

ψ[2] = randomITensor(inds(ψ[2]))

# ortho_lims(ψ) = 2:3
@show ortho_lims(ψ)
```
"""
function ortho_lims(ψ::AbstractMPS)
  return leftlim(ψ)+1:rightlim(ψ)-1
end

"""
    ITensors.set_ortho_lims!(::MPS/MPO, r::UnitRange{Int})

Sets the range of sites of the orthogonality center of the MPS/MPO.

This is not exported and is an advanced feature that should be used with
care. Setting the orthogonality limits wrong can lead to incorrect results
when using ITensor MPS/MPO functions.

If you are modifying an MPS/MPO and want the orthogonality limits to be
preserved, please see the `@preserve_ortho` macro.
"""
function set_ortho_lims!(ψ::AbstractMPS, r::UnitRange{Int})
  setleftlim!(ψ, first(r)-1)
  setrightlim!(ψ, last(r)+1)
  return ψ
end

isortho(m::AbstractMPS) = leftlim(m)+1 == rightlim(m)-1

function orthocenter(m::T) where {T<:AbstractMPS}
  !isortho(m) && error("$T has no well-defined orthogonality center")
  return leftlim(m)+1
end

getindex(M::AbstractMPS, n::Integer) =
  getindex(data(M), n)

lastindex(M::AbstractMPS) =
  lastindex(data(M))

"""
    @preserve_ortho

Specify that a block of code preserves the orthogonality limits of
an MPS/MPO that is being modified in-place. The first input is either
a single MPS/MPO or a tuple of the MPS/MPO whose orthogonality limits
should be preserved.

# Examples

```julia
s = siteinds("S=1/2", 4)

# Make random MPS with bond dimension 2
ψ₁ = randomMPS(s, "↑", 2)
ψ₂ = randomMPS(s, "↑", 2)
orthogonalize!(ψ₁, 1)
orthogonalize!(ψ₂, 1)

# ortho_lims(ψ₁) = 1:1
@show ortho_lims(ψ₁)

# ortho_lims(ψ₂) = 1:1
@show ortho_lims(ψ₂)

@preserve_ortho (ψ₁, ψ₂) begin
  ψ₁ .= addtags.(ψ₁, "x₁"; tags = "Link")
  ψ₂ .= addtags.(ψ₂, "x₂"; tags = "Link")
end

# ortho_lims(ψ₁) = 1:1
@show ortho_lims(ψ₁)

# ortho_lims(ψ₂) = 1:1
@show ortho_lims(ψ₂)
```
"""
macro preserve_ortho(ψ, block)
  quote
    if $(esc(ψ)) isa AbstractMPS
      local ortho_limsψ = ortho_lims($(esc(ψ)))
    else
      local ortho_limsψ = ortho_lims.($(esc(ψ)))
    end
    r = $(esc(block))
    if $(esc(ψ)) isa AbstractMPS
      set_ortho_lims!($(esc(ψ)), ortho_limsψ)
    else
      set_ortho_lims!.($(esc(ψ)), ortho_limsψ)
    end
    r
  end
end

function setindex!(M::AbstractMPS,
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

function setindex!(M::MPST, v::MPST, ::Colon) where {MPST <: AbstractMPS}
  setleftlim!(M, leftlim(v))
  setrightlim!(M, rightlim(v))
  data(M)[:] = data(v)
  return M
end

setindex!(M::AbstractMPS, v::Vector{<:ITensor}, ::Colon) =
  setindex!(M, MPS(v), :)

copy(m::AbstractMPS) = typeof(m)(copy(data(m)), leftlim(m), rightlim(m))

similar(m::AbstractMPS) = typeof(m)(similar(data(m)), 0, length(m))

deepcopy(m::AbstractMPS) =
  typeof(m)(deepcopy(data(m)), leftlim(m), rightlim(m))

eachindex(m::AbstractMPS) = 1:length(m)

iterate(M::AbstractMPS) = iterate(data(M))

iterate(M::AbstractMPS, state) = iterate(data(M), state)

"""
    linkind(M::MPS, j::Integer)

    linkind(M::MPO, j::Integer)

Get the link or bond Index connecting the
MPS or MPO tensor on site j to site j+1.

If there is no link Index, return `nothing`.
"""
function linkind(M::AbstractMPS, j::Integer)
  N = length(M)
  (j ≥ length(M) || j < 1) && return nothing
  return commonind(M[j], M[j+1])
end

linkinds(ψ::AbstractMPS) =
  [linkind(ψ, b) for b in 1:length(ψ)-1]

"""
    dense(::MPS/MPO)

Given an MPS (or MPO), return a new MPS (or MPO) 
having called `dense` on each ITensor to convert each
tensor to use dense storage and remove any QN or other
sparse structure information, if it is not dense already.
"""
function dense(ψ::AbstractMPS)
  ψ = copy(ψ)
  @preserve_ortho ψ ψ .= dense.(ψ)
  return ψ
end

"""
    unique_siteind(A::MPO, B::MPS, j::Integer)
    unique_siteind(A::MPO, B::MPO, j::Integer)

Get the site index of MPO `A` that is unique to `A` (not shared with MPS/MPO `B`).
"""
function unique_siteind(A::AbstractMPS, B::AbstractMPS, j::Integer)
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
    common_siteind(A::MPO, B::MPS, j::Integer)
    common_siteind(A::MPO, B::MPO, j::Integer)

Get the site index of MPO `A` that is shared with MPS/MPO `B`.
"""
function common_siteind(A::AbstractMPS, B::AbstractMPS, j::Integer)
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

keys(ψ::AbstractMPS) = keys(data(ψ))

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
    firstsiteind(M::Union{MPS,MPO}, j::Integer; kwargs...)
    siteind(::typeof(first), M::Union{MPS,MPO}, j::Integer; kwargs...)

Return the first site Index found on the MPS or MPO
(the first Index unique to the `j`th MPS/MPO tensor).

You can choose different filters, like prime level
and tags, with the `kwargs`.
"""
function siteind(::typeof(first), M::AbstractMPS, j::Integer;
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

firstsiteind(M::AbstractMPS, j::Integer; kwargs...) =
  siteind(first, M, j; kwargs...)

"""
    siteinds(M::Union{MPS, MPO}}, j::Integer; kwargs...)

Return the site Indices found of the MPO or MPO
at the site `j` as an IndexSet.

Optionally filter prime tags and prime levels with
keyword arguments like `plev` and `tags`.
"""
function siteinds(M::AbstractMPS, j::Integer; kwargs...)
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

function siteinds(::typeof(all), ψ::AbstractMPS, n::Integer; kwargs...)
  return siteinds(ψ, n; kwargs...)
end

siteinds(::typeof(first), ψ::AbstractMPS; kwargs...) =
  [siteind(first, ψ, j; kwargs...) for j in 1:length(ψ)]

siteinds(::typeof(only), ψ::AbstractMPS; kwargs...) =
  [siteind(only, ψ, j; kwargs...) for j in 1:length(ψ)]

siteinds(::typeof(all), ψ::AbstractMPS; kwargs...) =
  [siteinds(ψ, j; kwargs...) for j in 1:length(ψ)]

function replaceinds!(::typeof(linkinds), M::AbstractMPS,
                      l̃s::Vector{<:Index})
  for i in eachindex(M)[1:end-1]
    l = linkind(M, i)
    l̃ = l̃s[i]
    if !isnothing(l)
      @preserve_ortho M begin
        M[i] = replaceinds(M[i], l => l̃)
        M[i+1] = replaceinds(M[i+1], l => l̃)
      end
    end
  end
  return M
end

replaceinds(::typeof(linkinds), M::ITensors.AbstractMPS,
            l̃s::Vector{<:Index}) = replaceinds!(linkinds, copy(M), l̃s)

# TODO: change kwarg from `set_limits` to `preserve_ortho`
function map!(f::Function, M::AbstractMPS; set_limits::Bool = true)
  for i in eachindex(M)
    M[i, set_limits = set_limits] = f(M[i])
  end
  return M
end

# TODO: change kwarg from `set_limits` to `preserve_ortho`
Base.map(f::Function, M::AbstractMPS; set_limits::Bool = true) =
  map!(f, copy(M); set_limits = set_limits)

for fname in (:dag, :prime, :setprime, :noprime, :addtags, :removetags,
              :replacetags, :settags)
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

# Change to:
# map(f::Function, ::typeof(linkinds), M::AbstractMPS)
map_linkinds(f::Function, M::AbstractMPS) = map_linkinds!(f, copy(M))

# Change to:
# map!(f::Function, ::typeof(commoninds), ::typeof(siteinds),
#      M::AbstractMPS)
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

function hassameinds(::typeof(siteinds), M1::AbstractMPS, M2::AbstractMPS)
  length(M1) ≠ length(M2) && return false
  for n in 1:length(M1)
    !hassameinds(siteinds(all, M1, n), siteinds(all, M2, n)) && return false
  end
  return true
end

function hassamenuminds(::typeof(siteinds), M1::AbstractMPS, M2::AbstractMPS)
  length(M1) ≠ length(M2) && return false
  for n in 1:length(M1)
    length(siteinds(M1, n)) ≠ length(siteinds(M2, n)) && return false
  end
  return true
end

for fname in (:sim, :prime, :setprime, :noprime, :addtags, :removetags,
              :replacetags, :settags)
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
    linkdim(M::MPS, j::Integer)

    linkdim(M::MPO, j::Integer)

Get the dimension of the link or bond connecting the
MPS or MPO tensor on site j to site j+1.

If there is no link Index, return `nothing`.
"""
function linkdim(ψ::AbstractMPS, b::Integer)
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
  siteindsM1dag = siteinds(all, M1dag)
  siteindsM2 = siteinds(all, M2)
  if any(n -> length(n) > 1, siteindsM1dag) ||
     any(n -> length(n) > 1, siteindsM2) ||
     !hassamenuminds(siteinds, M1, M2)
    # If the MPS have more than one site Indices on any site or they don't have
    # the same number of site indices on each site, don't try to make the
    # indices match
    make_inds_match = false
  end
  if make_inds_match
    replace_siteinds!(M1dag, siteindsM2)
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
function dot(M1::MPST, M2::MPST; kwargs...) where {MPST <: AbstractMPS}
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
function logdot(M1::MPST, M2::MPST;
                kwargs...) where {MPST <: AbstractMPS}
  return _log_or_not_dot(M1, M2, true; kwargs...)
end

inner(M1::MPST, M2::MPST; kwargs...) where {MPST <: AbstractMPS} =
  dot(M1, M2; kwargs...)

loginner(M1::MPST, M2::MPST; kwargs...) where {MPST <: AbstractMPS} =
  logdot(M1, M2; kwargs...)

"""
    norm(A::MPS)

    norm(A::MPO)

Compute the norm of the MPS or MPO.

See also `lognorm`.
"""
norm(M::AbstractMPS) = sqrt(dot(M, M))

"""
    lognorm(A::MPS)

    lognorm(A::MPO)

Compute the logarithm of the norm of the MPS or MPO. 

This is useful for larger MPS/MPO that are not gauged, where in the limit of large numbers of sites the norm can diverge or approach zero.

See also `norm` and `loginner`/`logdot`.
"""
lognorm(M::AbstractMPS) = 0.5 * logdot(M, M)

function site_combiners(ψ::AbstractMPS)
  N = length(ψ)
  Cs = Vector{ITensor}(undef, N)
  for n in 1:N
    s = siteinds(all, ψ, n)
    Cs[n] = combiner(s; tags = commontags(s))
  end
  return Cs
end

# The maximum link dimensions when adding MPS/MPO
function _add_maxlinkdims(ψ⃗::AbstractMPS...)
  N = length(ψ⃗[1])
  maxdims = Vector{Int}(undef, N-1)
  for b in 1:N-1
    maxdims[b] = sum(ψ -> linkdim(ψ, b), ψ⃗)
  end
  return maxdims
end

"""
    +(A::MPS/MPO...; kwargs...)
    add(A::MPS/MPO...; kwargs...)

Add arbitrary numbers of MPS/MPO with each other, with some optional
truncation.

A cutoff of 1e-15 is used by default, and in general users should set their own cutoff for their particular application.

In the future we will give an interface for returning the truncation error.

# Examples

```julia
N = 10

s = siteinds("S=1/2", N; conserve_qns = true)

state = n -> isodd(n) ? "↑" : "↓"
ψ₁ = randomMPS(s, state, 2)
ψ₂ = randomMPS(s, state, 2)
ψ₃ = randomMPS(s, state, 2)

ψ = +(ψ₁, ψ₂; cutoff = 1e-8)

# Can use:
#
# ψ = ψ₁ + ψ₂
#
# but generally you want to set a custom `cutoff` and `maxdim`.

println()
@show inner(ψ, ψ)
@show inner(ψ₁, ψ₂) + inner(ψ₁, ψ₂) + inner(ψ₂, ψ₁) + inner(ψ₂, ψ₂)

# Computes ψ₁ + 2ψ₂
ψ = ψ₁ + 2ψ₂

println()
@show inner(ψ, ψ)
@show inner(ψ₁, ψ₁) + 2 * inner(ψ₁, ψ₂) + 2 * inner(ψ₂, ψ₁) + 4 * inner(ψ₂, ψ₂)

# Computes ψ₁ + 2ψ₂ + ψ₃
ψ = ψ₁ + 2ψ₂ + ψ₃

println()
@show inner(ψ, ψ)
@show inner(ψ₁, ψ₁) + 2 * inner(ψ₁, ψ₂) + inner(ψ₁, ψ₃) +
      2 * inner(ψ₂, ψ₁) + 4 * inner(ψ₂, ψ₂) + 2 * inner(ψ₂, ψ₃) +
      inner(ψ₃, ψ₁) + 2 * inner(ψ₃, ψ₂) + inner(ψ₃, ψ₃)
```
"""
function +(ψ⃗::MPST...;
           cutoff = 1e-15, kwargs...) where {MPST <: AbstractMPS}
  Nₘₚₛ = length(ψ⃗)

  @assert all(ψᵢ -> length(ψ⃗[1]) == length(ψᵢ), ψ⃗)

  N = length(ψ⃗[1])

  ψ⃗ = copy.(ψ⃗)

  X⃗ = site_combiners(ψ⃗[1])

  for ψᵢ in ψ⃗
    @preserve_ortho ψᵢ ψᵢ .*= X⃗
  end

  ψ⃗ = convert.(MPS, ψ⃗)

  s = siteinds(ψ⃗[1])

  ψ⃗ = orthogonalize.(ψ⃗, N)

  ψ = MPS(N)

  ρ⃗ₙ = [prime(ψᵢ[N], s[N]) * dag(ψᵢ[N]) for ψᵢ in ψ⃗]
  ρₙ = sum(ρ⃗ₙ)

  # Maximum theoretical link dimensions
  add_maxlinkdims = _add_maxlinkdims(ψ⃗...)

  C⃗ₙ = last.(ψ⃗)
  for n in reverse(2:N)
    Dₙ, Vₙ, spec = eigen(ρₙ; ishermitian = true,
                             tags = tags(linkind(ψ⃗[1], n-1)),
                             cutoff = cutoff,
                             maxdim = add_maxlinkdims[n-1],
                             kwargs...)
    lₙ₋₁ = commonind(Dₙ, Vₙ)

    # Update the total state
    ψ[n] = Vₙ

    # Compute the new density matrix
    C⃗ₙ₋₁ = [ψ⃗[i][n-1] * C⃗ₙ[i] * dag(Vₙ) for i in 1:Nₘₚₛ]
    C⃗ₙ₋₁′ = [prime(Cₙ₋₁, (s[n-1], lₙ₋₁)) for Cₙ₋₁ in C⃗ₙ₋₁]
    ρ⃗ₙ₋₁ = C⃗ₙ₋₁′ .* dag.(C⃗ₙ₋₁)
    ρₙ₋₁ = sum(ρ⃗ₙ₋₁)

    C⃗ₙ = C⃗ₙ₋₁
    ρₙ = ρₙ₋₁
  end

  ψ[1] = sum(C⃗ₙ)
  ψ .*= dag.(X⃗)

  set_ortho_lims!(ψ, 1:1)

  return convert(MPST, ψ)
end

add(ψ⃗::AbstractMPS...; kwargs...) = +(ψ⃗...; kwargs...)

-(ψ₁::AbstractMPS, ψ₂::AbstractMPS; kwargs...) =
  +(ψ₁, -ψ₂; kwargs...)

add(A::T, B::T;
    kwargs...) where {T <: AbstractMPS} = +(A, B; kwargs...)

"""
    sum(A::Vector{MPS}; kwargs...)

    sum(A::Vector{MPO}; kwargs...)

Add multiple MPS/MPO with each other, with some optional
truncation.
"""
function Base.sum(ψ⃗::Vector{T};
                  kwargs...) where {T <: AbstractMPS}
  length(ψ⃗) == 0 && return T()
  length(ψ⃗) == 1 && return A[1]
  return +(ψ⃗...; kwargs...)
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
function truncate!(M::AbstractMPS; kwargs...)
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

contract(A::AbstractMPS, B::AbstractMPS; kwargs...) =
  *(A, B; kwargs...)

"""
    α::Number * ψ::MPS/MPO

Scales the MPS or MPO by the provided number.

Currently, this works by scaling one of the sites within the orthogonality limits.
"""
function (α::Number * ψ::AbstractMPS)
  limsψ = ortho_lims(ψ)
  n = first(limsψ)
  αψ = copy(ψ)
  αψ[n] = α * ψ[n]
  return αψ
end

(ψ::AbstractMPS * α::Number) = α * ψ

-(ψ::AbstractMPS) = -1 * ψ

"""
    setindex!(::Union{MPS, MPO}, ::Union{MPS, MPO},
              r::UnitRange{Int64})

Sets a contiguous range of MPS/MPO tensors
"""
function setindex!(ψ::MPST, ϕ::MPST,
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

_isodd_fermionic_parity(s::Index, ::Integer) = false

function _isodd_fermionic_parity(s::QNIndex, n::Integer)
  qn_n = qn(space(s)[n])
  fermionic_qn_pos = findfirst(q -> isfermionic(q), qn_n)
  isnothing(fermionic_qn_pos) && return false
  return isodd(val(qn_n[fermionic_qn_pos]))
end

function _fermionic_swap(s1::Index, s2::Index)
  T = ITensor(s1', s2', dag(s1), dag(s2))
  for b in nzblocks(T)
    dval = 1.0
    # Must be a diagonal block
    ((b[1] ≠ b[3]) || (b[2] ≠ b[4])) && continue
    n1, n2 = b[1], b[2]
    if _isodd_fermionic_parity(s1, n1) &&
       _isodd_fermionic_parity(s2, n2)
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
function setindex!(ψ::MPST, A::ITensor, r::UnitRange{Int};
                   orthocenter::Integer = last(r),
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

setindex!(ψ::MPST, A::ITensor, r::UnitRange{Int}, args::Pair{Symbol}...;
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
- `orthocenter::Integer = length(sites)`: the desired final orthogonality center of the output MPS/MPO.
- `cutoff`: the desired truncation error at each link.
- `maxdim`: the maximum link dimension.
"""
function (::Type{MPST})(A::ITensor, sites;
                        leftinds = nothing,
                        orthocenter::Integer = length(sites),
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
    swapbondsites(ψ::Union{MPS, MPO}, b::Integer; kwargs...)

Swap the sites `b` and `b+1`.
"""
function swapbondsites(ψ::AbstractMPS, b::Integer; kwargs...)
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
                  orthocenter::Integer = last(n1n2),
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
    apply([...])

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
    product(As::Vector{<:ITensor}, M::Union{MPS, MPO}; <keyword arguments>)
    apply([...])

Apply the ITensors `As` to the MPS or MPO `M`, treating them as gates or matrices from pairs of prime or unprimed indices.

# Examples

Apply one-site gates to an MPS:
```julia
N = 3

ITensors.op(::OpName"σx", ::SiteType"S=1/2", s::Index) =
  2*op("Sx", s)

ITensors.op(::OpName"σz", ::SiteType"S=1/2", s::Index) =
  2*op("Sz", s)

# Make the operator list.
os = [("σx", n) for n in 1:N]
append!(os, [("σz", n) for n in 1:N])

@show os

s = siteinds("S=1/2", N)
gates = ops(os, s)

# Starting state |↑↑↑⟩
ψ0 = productMPS(s, "↑")

# Apply the gates.
ψ = apply(gates, ψ0; cutoff = 1e-15)

# Test against exact (full) wavefunction
prodψ = apply(gates, prod(ψ0))
@show prod(ψ) ≈ prodψ

# The result is:
# σz₃ σz₂ σz₁ σx₃ σx₂ σx₁ |↑↑↑⟩ = -|↓↓↓⟩
@show inner(ψ, productMPS(s, "↓")) == -1
```
Apply nonlocal two-site gates and one-site gates to an MPS:
```julia
# 2-site gate
function ITensors.op(::OpName"CX", ::SiteType"S=1/2", s1::Index, s2::Index)
  mat = [1 0 0 0
         0 1 0 0
         0 0 0 1
         0 0 1 0]
  return itensor(mat, s2', s1', s2, s1)
end

os = [("CX", 1, 3), ("σz", 3)]

@show os

# Start with the state |↓↑↑⟩
ψ0 = productMPS(s, n -> n == 1 ? "↓" : "↑")

# The result is:
# σz₃ CX₁₃ |↓↑↑⟩ = -|↓↑↓⟩
ψ = apply(ops(os, s), ψ0; cutoff = 1e-15)
@show inner(ψ, productMPS(s, n -> n == 1 || n == 3 ? "↓" : "↑")) == -1
```
Perform TEBD-like time evolution:
```julia
# Define the nearest neighbor term `S⋅S` for the Heisenberg model
function ITensors.op(::OpName"expS⋅S", ::SiteType"S=1/2",
                     s1::Index, s2::Index; τ::Number)
  O = 0.5 * op("S+", s1) * op("S-", s2) +
      0.5 * op("S-", s1) * op("S+", s2) +
            op("Sz", s1) * op("Sz", s2)
  return exp(τ * O)
end

τ = -0.1im
os = [("expS⋅S", (1, 2), (τ = τ,)),
      ("expS⋅S", (2, 3), (τ = τ,))]
ψ0 = productMPS(s, n -> n == 1 ? "↓" : "↑")
expτH = ops(os, s)
ψτ = apply(expτH, ψ0)
```
"""
function product(As::Vector{ <: ITensor}, ψ::AbstractMPS;
                 move_sites_back::Bool = true, kwargs...)
  Aψ = ψ
  for A in As
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

#
# QN functions
#

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

"""
    splitblocks[!](::typeof(linkinds), M::AbstractMPS; tol = 0)

Split the QN blocks of the links of the MPS or MPO into dimension 1 blocks. Then, only keep the blocks with `norm(b) > tol`.

This can make the ITensors of the MPS/MPO more sparse, and is particularly helpful as a preprocessing step on a local Hamiltonian MPO for DMRG.
"""
function splitblocks!(::typeof(linkinds), M::AbstractMPS; tol = 0)
  for i in eachindex(M)[1:end-1]
    l = linkind(M, i)
    if !isnothing(l)
      @preserve_ortho M begin
        M[i] = splitblocks(M[i], l)
        M[i+1] = splitblocks(M[i+1], l)
      end
    end
  end
  return M
end

splitblocks(::typeof(linkinds), M::AbstractMPS; tol = 0) =
  splitblocks!(linkinds, copy(M); tol = 0)

#
# Broadcasting
#

BroadcastStyle(MPST::Type{<:AbstractMPS}) = Style{MPST}()

copyto!(ψ::AbstractMPS, b::Broadcasted) = copyto!(data(ψ), b)

#
# Printing functions
#

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

#
# Old code for adding MPS/MPO
#

#function plussers(::Type{T},
#                  left_ind::Index,
#                  right_ind::Index,
#                  sum_ind::Index) where {T<:Array}
#  total_dim    = dim(left_ind) + dim(right_ind)
#  total_dim    = max(total_dim, 1)
#  # TODO: I am not sure if we should be using delta
#  # tensors for this purpose? I think we should consider
#  # not allowing them to be made with different index sizes
#  #left_tensor  = δ(left_ind, sum_ind)
#  left_tensor  = diagITensor(1.0,left_ind, sum_ind)
#  right_tensor = ITensor(right_ind, sum_ind)
#  for i in 1:dim(right_ind)
#    right_tensor[right_ind(i), sum_ind(dim(left_ind) + i)] = 1
#  end
#  return left_tensor, right_tensor
#end
#
#function Base.:+(A::T, B::T; kwargs...) where {T <: AbstractMPS}
#  A = copy(A)
#  B = copy(B)
#  N = length(A)
#  length(B) != N && throw(DimensionMismatch("lengths of MPOs A ($N) and B ($(length(B))) do not match"))
#  orthogonalize!(A, 1; kwargs...)
#  orthogonalize!(B, 1; kwargs...)
#  C = similar(A)
#  rand_plev = 13124
#  lAs = [linkind(A, i) for i in 1:N-1]
#  prime!(A, rand_plev, "Link")
#
#  first  = Vector{ITensor{2}}(undef,N-1)
#  second = Vector{ITensor{2}}(undef,N-1)
#  for i in 1:N-1
#    lA = linkind(A, i)
#    lB = linkind(B, i)
#    r  = Index(dim(lA) + dim(lB), tags(lA))
#    f, s = plussers(typeof(data(A[1])), lA, lB, r)
#    first[i]  = f
#    second[i] = s
#  end
#  C[1] = A[1] * first[1] + B[1] * second[1]
#  for i in 2:N-1
#      C[i] = dag(first[i-1]) * A[i] * first[i] + dag(second[i-1]) * B[i] * second[i]
#  end
#  C[N] = dag(first[N-1]) * A[N] + dag(second[N-1]) * B[N]
#  prime!(C, -rand_plev, "Link")
#  truncate!(C; kwargs...)
#  return C
#end

