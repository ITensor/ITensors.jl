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

function promote_itensor_eltype(m::Vector{ITensor})
  T = isassigned(m, 1) ? eltype(m[1]) : Number
  for n in 2:length(m)
    Tn = isassigned(m, n) ? eltype(m[n]) : Number
    T = promote_type(T, Tn)
  end
  return T
end

function LinearAlgebra.promote_leaf_eltypes(m::Vector{ITensor})
  return promote_itensor_eltype(m)
end

function LinearAlgebra.promote_leaf_eltypes(m::AbstractMPS)
  return LinearAlgebra.promote_leaf_eltypes(data(m))
end

"""
    promote_itensor_eltype(m::MPS)
    promote_itensor_eltype(m::MPO)

Return the promotion of the elements type of the
tensors in the MPS or MPO. For example,
if all tensors have type `Float64` then
return `Float64`. But if one or more tensors
have type `ComplexF64`, return `ComplexF64`.
"""
promote_itensor_eltype(m::AbstractMPS) = LinearAlgebra.promote_leaf_eltypes(m)

scalartype(m::AbstractMPS) = LinearAlgebra.promote_leaf_eltypes(m)
scalartype(m::Array{ITensor}) = LinearAlgebra.promote_leaf_eltypes(m)
scalartype(m::Array{<:Array{ITensor}}) = LinearAlgebra.promote_leaf_eltypes(m)

"""
    eltype(m::MPS)
    eltype(m::MPO)

The element type of the MPS/MPO. Always returns `ITensor`.

For the element type of the ITensors of the MPS/MPO,
use `promote_itensor_eltype`.
"""
eltype(::AbstractMPS) = ITensor

complex(ψ::AbstractMPS) = complex.(ψ)
real(ψ::AbstractMPS) = real.(ψ)
imag(ψ::AbstractMPS) = imag.(ψ)
conj(ψ::AbstractMPS) = conj.(ψ)

function convert_leaf_eltype(eltype::Type, ψ::AbstractMPS)
  return map(ψᵢ -> convert_leaf_eltype(eltype, ψᵢ), ψ; set_limits=false)
end

"""
    ITensors.data(::MPS/MPO)

Returns a view of the Vector storage of an MPS/MPO.

This is not exported and mostly for internal usage, please let us
know if there is functionality not available for MPS/MPO you would like.
"""
data(m::AbstractMPS) = m.data

contract(ψ::AbstractMPS) = contract(data(ψ))

leftlim(m::AbstractMPS) = m.llim

rightlim(m::AbstractMPS) = m.rlim

function setleftlim!(m::AbstractMPS, new_ll::Integer)
  return m.llim = new_ll
end

function setrightlim!(m::AbstractMPS, new_rl::Integer)
  return m.rlim = new_rl
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
  return (leftlim(ψ) + 1):(rightlim(ψ) - 1)
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
  setleftlim!(ψ, first(r) - 1)
  setrightlim!(ψ, last(r) + 1)
  return ψ
end

function set_ortho_lims(ψ::AbstractMPS, r::UnitRange{Int})
  return set_ortho_lims!(copy(ψ), r)
end

reset_ortho_lims!(ψ::AbstractMPS) = set_ortho_lims!(ψ, 1:length(ψ))

isortho(m::AbstractMPS) = leftlim(m) + 1 == rightlim(m) - 1

# Could also define as `only(ortho_lims)`
function orthocenter(m::AbstractMPS)
  !isortho(m) && error(
    "$(typeof(m)) has no well-defined orthogonality center, orthogonality center is on the range $(ortho_lims(m)).",
  )
  return leftlim(m) + 1
end

getindex(M::AbstractMPS, n) = getindex(data(M), n)

isassigned(M::AbstractMPS, n) = isassigned(data(M), n)

lastindex(M::AbstractMPS) = lastindex(data(M))

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

function setindex!(M::AbstractMPS, T::ITensor, n::Integer; set_limits::Bool=true)
  if set_limits
    (n <= leftlim(M)) && setleftlim!(M, n - 1)
    (n >= rightlim(M)) && setrightlim!(M, n + 1)
  end
  data(M)[n] = T
  return M
end

function setindex!(M::MPST, v::MPST, ::Colon) where {MPST<:AbstractMPS}
  setleftlim!(M, leftlim(v))
  setrightlim!(M, rightlim(v))
  data(M)[:] = data(v)
  return M
end

setindex!(M::AbstractMPS, v::Vector{<:ITensor}, ::Colon) = setindex!(M, MPS(v), :)

"""
    copy(::MPS)
    copy(::MPO)

Make a shallow copy of an MPS or MPO. By shallow copy, it means that a new MPS/MPO
is returned, but the data of the tensors are still shared between the returned MPS/MPO
and the original MPS/MPO.

Therefore, replacing an entire tensor of the returned MPS/MPO will not modify the input MPS/MPO,
but modifying the data of the returned MPS/MPO will modify the input MPS/MPO.

Use [`deepcopy`](@ref) for an alternative that copies the ITensors as well.

# Examples
```julia
julia> using ITensors

julia> s = siteinds("S=1/2", 3);

julia> M1 = randomMPS(s; linkdims=3);

julia> norm(M1)
0.9999999999999999

julia> M2 = copy(M1);

julia> M2[1] *= 2;

julia> norm(M1)
0.9999999999999999

julia> norm(M2)
1.9999999999999998

julia> M3 = copy(M1);

julia> M3[1] .*= 3; # Modifies the tensor data

julia> norm(M1)
3.0000000000000004

julia> norm(M3)
3.0000000000000004
```
"""
copy(m::AbstractMPS) = typeof(m)(copy(data(m)), leftlim(m), rightlim(m))

similar(m::AbstractMPS) = typeof(m)(similar(data(m)), 0, length(m))

"""
    deepcopy(::MPS)
    deepcopy(::MPO)

Make a deep copy of an MPS or MPO. By deep copy, it means that a new MPS/MPO
is returned that doesn't share any data with the input MPS/MPO.

Therefore, modifying the resulting MPS/MPO will note modify the original MPS/MPO.

Use [`copy`](@ref) for an alternative that performs a shallow copy that avoids
copying the ITensor data.

# Examples
```julia
julia> using ITensors

julia> s = siteinds("S=1/2", 3);

julia> M1 = randomMPS(s; linkdims=3);

julia> norm(M1)
1.0

julia> M2 = deepcopy(M1);

julia> M2[1] .*= 2; # Modifies the tensor data

julia> norm(M1)
1.0

julia> norm(M2)
2.0

julia> M3 = copy(M1);

julia> M3[1] .*= 3; # Modifies the tensor data

julia> norm(M1)
3.0

julia> norm(M3)
3.0
```
"""
deepcopy(m::AbstractMPS) = typeof(m)(copy.(data(m)), leftlim(m), rightlim(m))

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
  return commonind(M[j], M[j + 1])
end

"""
    linkinds(M::MPS, j::Integer)
    linkinds(M::MPO, j::Integer)

Get all of the link or bond Indices connecting the
MPS or MPO tensor on site j to site j+1.
"""
function linkinds(M::AbstractMPS, j::Integer)
  N = length(M)
  (j ≥ length(M) || j < 1) && return IndexSet()
  return commoninds(M[j], M[j + 1])
end

linkinds(ψ::AbstractMPS) = [linkind(ψ, b) for b in 1:(length(ψ) - 1)]

function linkinds(::typeof(all), ψ::AbstractMPS)
  return IndexSet[linkinds(ψ, b) for b in 1:(length(ψ) - 1)]
end

#
# Internal tools for checking for default link tags
#

"""
    ITensors.defaultlinktags(b::Integer)

Default link tags for link index connecting sites `b` to `b+1`.
"""
defaultlinktags(b::Integer) = TagSet("Link,l=$b")

"""
    ITensors.hasdefaultlinktags(ψ::MPS/MPO)

Return true if the MPS/MPO has default link tags.
"""
function hasdefaultlinktags(ψ::AbstractMPS)
  ls = linkinds(all, ψ)
  for (b, lb) in enumerate(ls)
    if length(lb) ≠ 1 || tags(only(lb)) ≠ defaultlinktags(b)
      return false
    end
  end
  return true
end

"""
    ITensors.eachlinkinds(ψ::MPS/MPO)

Return an iterator over each of the sets of link indices of the MPS/MPO.
"""
eachlinkinds(ψ::AbstractMPS) = (linkinds(ψ, n) for n in eachindex(ψ)[1:(end - 1)])

"""
    ITensors.eachsiteinds(ψ::MPS/MPO)

Return an iterator over each of the sets of site indices of the MPS/MPO.
"""
eachsiteinds(ψ::AbstractMPS) = (siteinds(ψ, n) for n in eachindex(ψ))

"""
    ITensors.hasnolinkinds(ψ::MPS/MPO)

Return true if the MPS/MPO has no link indices.
"""
function hasnolinkinds(ψ::AbstractMPS)
  for l in eachlinkinds(ψ)
    if length(l) > 0
      return false
    end
  end
  return true
end

"""
    ITensors.insertlinkinds(ψ::MPS/MPO)

If any link indices are missing, insert default ones.
"""
function insertlinkinds(ψ::AbstractMPS)
  ψ = copy(ψ)
  space = hasqns(ψ) ? [QN() => 1] : 1
  linkind(b::Integer) = Index(space; tags=defaultlinktags(b))
  for b in 1:(length(ψ) - 1)
    if length(linkinds(ψ, b)) == 0
      lb = ITensor(1, linkind(b))
      @preserve_ortho ψ begin
        ψ[b] = ψ[b] * lb
        ψ[b + 1] = ψ[b + 1] * dag(lb)
      end
    end
  end
  return ψ
end

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
    siteinds(uniqueinds, A::MPO, B::MPS, j::Integer; kwargs...)
    siteinds(uniqueind, A::MPO, B::MPS, j::Integer; kwargs...)

Get the site index (or indices) of MPO `A` that is unique to `A` (not shared with MPS/MPO `B`).
"""
function siteinds(
  f::Union{typeof(uniqueinds),typeof(uniqueind)},
  A::AbstractMPS,
  B::AbstractMPS,
  j::Integer;
  kwargs...,
)
  N = length(A)
  N == 1 && return f(A[j], B[j]; kwargs...)
  j == 1 && return f(A[j], A[j + 1], B[j]; kwargs...)
  j == N && return f(A[j], A[j - 1], B[j]; kwargs...)
  return f(A[j], A[j - 1], A[j + 1], B[j]; kwargs...)
end

"""
    siteinds(uniqueinds, A::MPO, B::MPS)
    siteinds(uniqueind, A::MPO, B::MPO)

Get the site indices of MPO `A` that are unique to `A` (not shared with MPS/MPO `B`), as a `Vector{<:Index}`.
"""
function siteinds(
  f::Union{typeof(uniqueinds),typeof(uniqueind)}, A::AbstractMPS, B::AbstractMPS; kwargs...
)
  return [siteinds(f, A, B, j; kwargs...) for j in eachindex(A)]
end

"""
    siteinds(commoninds, A::MPO, B::MPS, j::Integer; kwargs...)
    siteinds(commonind, A::MPO, B::MPO, j::Integer; kwargs...)

Get the site index (or indices) of  the `j`th MPO tensor of `A` that is shared with MPS/MPO `B`.
"""
function siteinds(
  f::Union{typeof(commoninds),typeof(commonind)},
  A::AbstractMPS,
  B::AbstractMPS,
  j::Integer;
  kwargs...,
)
  return f(A[j], B[j]; kwargs...)
end

"""
    siteinds(commoninds, A::MPO, B::MPS; kwargs...)
    siteinds(commonind, A::MPO, B::MPO; kwargs...)

Get a vector of the site index (or indices) of MPO `A` that is shared with MPS/MPO `B`.
"""
function siteinds(
  f::Union{typeof(commoninds),typeof(commonind)}, A::AbstractMPS, B::AbstractMPS; kwargs...
)
  return [siteinds(f, A, B, j) for j in eachindex(A)]
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

#
# TODO: Maybe make:
# findall(f::Function, siteindsM::Tuple{typeof(siteinds), ::AbstractMPS})
# findall(siteindsM::Tuple{typeof(siteinds), <:AbstractMPS}, is) =
#   findall(hassameinds(is), siteindsM)
#
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
findsites(ψ::AbstractMPS, is) = findall(hascommoninds(is), ψ)

findsites(ψ::AbstractMPS, s::Index) = findsites(ψ, IndexSet(s))

#
# TODO: Maybe make:
# findfirst(f::Function, siteindsM::Tuple{typeof(siteinds), ::AbstractMPS})
# findfirst(siteindsM::Tuple{typeof(siteinds), <:AbstractMPS}, is) =
#   findfirst(hassameinds(is), siteindsM)
#
"""
    findfirstsiteind(M::MPS, i::Index)
    findfirstsiteind(M::MPO, i::Index)

Return the first site of the MPS or MPO that has the
site index `i`.
"""
findfirstsiteind(ψ::AbstractMPS, s::Index) = findfirst(hasind(s), ψ)

# TODO: depracate in favor of findsite.
"""
    findfirstsiteinds(M::MPS, is)
    findfirstsiteinds(M::MPO, is)

Return the first site of the MPS or MPO that has the
site indices `is`.
"""
findfirstsiteinds(ψ::AbstractMPS, s) = findfirst(hasinds(s), ψ)

"""
    siteind(::typeof(first), M::Union{MPS,MPO}, j::Integer; kwargs...)

Return the first site Index found on the MPS or MPO
(the first Index unique to the `j`th MPS/MPO tensor).

You can choose different filters, like prime level
and tags, with the `kwargs`.
"""
function siteind(::typeof(first), M::AbstractMPS, j::Integer; kwargs...)
  N = length(M)
  (N == 1) && return firstind(M[1]; kwargs...)
  if j == 1
    si = uniqueind(M[j], M[j + 1]; kwargs...)
  elseif j == N
    si = uniqueind(M[j], M[j - 1]; kwargs...)
  else
    si = uniqueind(M[j], M[j - 1], M[j + 1]; kwargs...)
  end
  return si
end

"""
    siteinds(M::Union{MPS, MPO}}, j::Integer; kwargs...)

Return the site Indices found of the MPO or MPO
at the site `j` as an IndexSet.

Optionally filter prime tags and prime levels with
keyword arguments like `plev` and `tags`.
"""
function siteinds(M::AbstractMPS, j::Integer; kwargs...)
  N = length(M)
  (N == 1) && return inds(M[1]; kwargs...)
  if j == 1
    si = uniqueinds(M[j], M[j + 1]; kwargs...)
  elseif j == N
    si = uniqueinds(M[j], M[j - 1]; kwargs...)
  else
    si = uniqueinds(M[j], M[j - 1], M[j + 1]; kwargs...)
  end
  return si
end

function siteinds(::typeof(all), ψ::AbstractMPS, n::Integer; kwargs...)
  return siteinds(ψ, n; kwargs...)
end

function siteinds(::typeof(first), ψ::AbstractMPS; kwargs...)
  return [siteind(first, ψ, j; kwargs...) for j in 1:length(ψ)]
end

function siteinds(::typeof(only), ψ::AbstractMPS; kwargs...)
  return [siteind(only, ψ, j; kwargs...) for j in 1:length(ψ)]
end

function siteinds(::typeof(all), ψ::AbstractMPS; kwargs...)
  return [siteinds(ψ, j; kwargs...) for j in 1:length(ψ)]
end

function replaceinds!(::typeof(linkinds), M::AbstractMPS, l̃s::Vector{<:Index})
  for i in eachindex(M)[1:(end - 1)]
    l = linkind(M, i)
    l̃ = l̃s[i]
    if !isnothing(l)
      @preserve_ortho M begin
        M[i] = replaceinds(M[i], l => l̃)
        M[i + 1] = replaceinds(M[i + 1], l => l̃)
      end
    end
  end
  return M
end

function replaceinds(::typeof(linkinds), M::AbstractMPS, l̃s::Vector{<:Index})
  return replaceinds!(linkinds, copy(M), l̃s)
end

# TODO: change kwarg from `set_limits` to `preserve_ortho`
function map!(f::Function, M::AbstractMPS; set_limits::Bool=true)
  for i in eachindex(M)
    M[i, set_limits=set_limits] = f(M[i])
  end
  return M
end

# TODO: change kwarg from `set_limits` to `preserve_ortho`
function map(f::Function, M::AbstractMPS; set_limits::Bool=true)
  return map!(f, copy(M); set_limits=set_limits)
end

for fname in (
  :dag,
  :prime,
  :setprime,
  :noprime,
  :swapprime,
  :replaceprime,
  :addtags,
  :removetags,
  :replacetags,
  :settags,
)
  fname! = Symbol(fname, :!)

  @eval begin
    """
        $($fname)[!](M::MPS, args...; kwargs...)
        $($fname)[!](M::MPO, args...; kwargs...)

    Apply $($fname) to all ITensors of an MPS/MPO, returning a new MPS/MPO.

    The ITensors of the MPS/MPO will be a view of the storage of the original ITensors. Alternatively apply the function in-place.
    """
    function $fname(M::AbstractMPS, args...; set_limits::Bool=false, kwargs...)
      return map(m -> $fname(m, args...; kwargs...), M; set_limits=set_limits)
    end

    function $(fname!)(M::AbstractMPS, args...; set_limits::Bool=false, kwargs...)
      return map!(m -> $fname(m, args...; kwargs...), M; set_limits=set_limits)
    end
  end
end

adjoint(M::AbstractMPS) = prime(M)

function hascommoninds(::typeof(siteinds), A::AbstractMPS, B::AbstractMPS)
  N = length(A)
  for n in 1:N
    !hascommoninds(siteinds(A, n), siteinds(B, n)) && return false
  end
  return true
end

function check_hascommoninds(::typeof(siteinds), A::AbstractMPS, B::AbstractMPS)
  N = length(A)
  if length(B) ≠ N
    throw(
      DimensionMismatch(
        "$(typeof(A)) and $(typeof(B)) have mismatched lengths $N and $(length(B))."
      ),
    )
  end
  for n in 1:N
    !hascommoninds(siteinds(A, n), siteinds(B, n)) && error(
      "$(typeof(A)) A and $(typeof(B)) B must share site indices. On site $n, A has site indices $(siteinds(A, n)) while B has site indices $(siteinds(B, n)).",
    )
  end
  return nothing
end

function map!(f::Function, ::typeof(linkinds), M::AbstractMPS)
  for i in eachindex(M)[1:(end - 1)]
    l = linkinds(M, i)
    if !isempty(l)
      l̃ = f(l)
      @preserve_ortho M begin
        M[i] = replaceinds(M[i], l, l̃)
        M[i + 1] = replaceinds(M[i + 1], l, l̃)
      end
    end
  end
  return M
end

map(f::Function, ::typeof(linkinds), M::AbstractMPS) = map!(f, linkinds, copy(M))

function map!(f::Function, ::typeof(siteinds), M::AbstractMPS)
  for i in eachindex(M)
    s = siteinds(M, i)
    if !isempty(s)
      @preserve_ortho M begin
        M[i] = replaceinds(M[i], s, f(s))
      end
    end
  end
  return M
end

map(f::Function, ::typeof(siteinds), M::AbstractMPS) = map!(f, siteinds, copy(M))

function map!(
  f::Function, ::typeof(siteinds), ::typeof(commoninds), M1::AbstractMPS, M2::AbstractMPS
)
  length(M1) != length(M2) && error("MPOs/MPSs must be the same length")
  for i in eachindex(M1)
    s = siteinds(commoninds, M1, M2, i)
    if !isempty(s)
      s̃ = f(s)
      @preserve_ortho (M1, M2) begin
        M1[i] = replaceinds(M1[i], s .=> s̃)
        M2[i] = replaceinds(M2[i], s .=> s̃)
      end
    end
  end
  return M1, M2
end

function map!(
  f::Function, ::typeof(siteinds), ::typeof(uniqueinds), M1::AbstractMPS, M2::AbstractMPS
)
  length(M1) != length(M2) && error("MPOs/MPSs must be the same length")
  for i in eachindex(M1)
    s = siteinds(uniqueinds, M1, M2, i)
    if !isempty(s)
      @preserve_ortho M1 begin
        M1[i] = replaceinds(M1[i], s .=> f(s))
      end
    end
  end
  return M1
end

function map(
  f::Function,
  ffilter::typeof(siteinds),
  fsubset::Union{typeof(commoninds),typeof(uniqueinds)},
  M1::AbstractMPS,
  M2::AbstractMPS,
)
  return map!(f, ffilter, fsubset, copy(M1), copy(M2))
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

for fname in
    (:sim, :prime, :setprime, :noprime, :addtags, :removetags, :replacetags, :settags)
  fname! = Symbol(fname, :!)
  @eval begin
    """
        $($fname)[!](linkinds, M::MPS, args...; kwargs...)
        $($fname)[!](linkinds, M::MPO, args...; kwargs...)

    Apply $($fname) to all link indices of an MPS/MPO, returning a new MPS/MPO.

    The ITensors of the MPS/MPO will be a view of the storage of the original ITensors.
    """
    function $fname(ffilter::typeof(linkinds), M::AbstractMPS, args...; kwargs...)
      return map(i -> $fname(i, args...; kwargs...), ffilter, M)
    end

    function $(fname!)(ffilter::typeof(linkinds), M::AbstractMPS, args...; kwargs...)
      return map!(i -> $fname(i, args...; kwargs...), ffilter, M)
    end

    """
        $($fname)[!](siteinds, M::MPS, args...; kwargs...)
        $($fname)[!](siteinds, M::MPO, args...; kwargs...)

    Apply $($fname) to all site indices of an MPS/MPO, returning a new MPS/MPO.

    The ITensors of the MPS/MPO will be a view of the storage of the original ITensors.
    """
    function $fname(ffilter::typeof(siteinds), M::AbstractMPS, args...; kwargs...)
      return map(i -> $fname(i, args...; kwargs...), ffilter, M)
    end

    function $(fname!)(ffilter::typeof(siteinds), M::AbstractMPS, args...; kwargs...)
      return map!(i -> $fname(i, args...; kwargs...), ffilter, M)
    end

    """
        $($fname)[!](siteinds, commoninds, M1::MPO, M2::MPS, args...; kwargs...)
        $($fname)[!](siteinds, commoninds, M1::MPO, M2::MPO, args...; kwargs...)

    Apply $($fname) to the site indices that are shared by `M1` and `M2`.

    Returns new MPSs/MPOs. The ITensors of the MPSs/MPOs will be a view of the storage of the original ITensors.
    """
    function $fname(
      ffilter::typeof(siteinds),
      fsubset::typeof(commoninds),
      M1::AbstractMPS,
      M2::AbstractMPS,
      args...;
      kwargs...,
    )
      return map(i -> $fname(i, args...; kwargs...), ffilter, fsubset, M1, M2)
    end

    function $(fname!)(
      ffilter::typeof(siteinds),
      fsubset::typeof(commoninds),
      M1::AbstractMPS,
      M2::AbstractMPS,
      args...;
      kwargs...,
    )
      return map!(i -> $fname(i, args...; kwargs...), ffilter, fsubset, M1, M2)
    end

    """
        $($fname)[!](siteinds, uniqueinds, M1::MPO, M2::MPS, args...; kwargs...)

    Apply $($fname) to the site indices of `M1` that are not shared with `M2`. Returns new MPSs/MPOs.

    The ITensors of the MPSs/MPOs will be a view of the storage of the original ITensors.
    """
    function $fname(
      ffilter::typeof(siteinds),
      fsubset::typeof(uniqueinds),
      M1::AbstractMPS,
      M2::AbstractMPS,
      args...;
      kwargs...,
    )
      return map(i -> $fname(i, args...; kwargs...), ffilter, fsubset, M1, M2)
    end

    function $(fname!)(
      ffilter::typeof(siteinds),
      fsubset::typeof(uniqueinds),
      M1::AbstractMPS,
      M2::AbstractMPS,
      args...;
      kwargs...,
    )
      return map!(i -> $fname(i, args...; kwargs...), ffilter, fsubset, M1, M2)
    end
  end
end

"""
    maxlinkdim(M::MPS)
    maxlinkdim(M::MPO)

Get the maximum link dimension of the MPS or MPO.

The minimum this will return is `1`, even if there
are no link indices.
"""
function maxlinkdim(M::AbstractMPS)
  md = 1
  for b in eachindex(M)[1:(end - 1)]
    l = linkind(M, b)
    linkdim = isnothing(l) ? 1 : dim(l)
    md = max(md, linkdim)
  end
  return md
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

linkdims(ψ::AbstractMPS) = [linkdim(ψ, b) for b in 1:(length(ψ) - 1)]

function inner_mps_mps_deprecation_warning()
  return """
 Calling `inner(x::MPS, y::MPS)` where the site indices of the `MPS` `x` and `y`
 don't match is deprecated as of ITensor v0.3 and will result in an error in ITensor
v0.4. Likely you are attempting to take the inner product of MPS that have site indices
with mismatched prime levels. The most common cause of this is something like the following:

 ```julia
 s = siteinds("S=1/2")
 psi = randomMPS(s)
 H = MPO(s, "Id")
 Hpsi = contract(H, psi; cutoff=1e-8) # or `Hpsi = *(H, psi; cutoff=1e-8)`
 inner(psi, Hpsi)
 ```

 `psi` has the Index structure `-s-(psi)` and `H` has the Index structure
 `-s'-(H)-s-`, so the contraction follows as: `-s'-(H)-s-(psi) ≈ -s'-(Hpsi)`.
 Then, the prime levels of `Hpsi` and `psi` don't match in `inner(psi, Hpsi)`.

 There are a few ways to fix this. You can simply change:

 ```julia
 inner(psi, Hpsi)
 ```

 to:

 ```julia
 inner(psi', Hpsi)
 ```

 in which case both `psi'` and `Hpsi` have primed site indices. Alternatively,
 you can use the `apply` function instead of the `contract` function, which
 calls `contract` and unprimes the resulting MPS:

 ```julia
 Hpsi = apply(H, psi; cutoff=1e-8) # or `Hpsi = H(psi; cutoff=1e-8)`
 inner(psi, Hpsi)
 ```

 Finally, if you only compute `Hpsi` to pass to the `inner` function, consider using:

 ```julia
 inner(psi', H, psi)
 ```

 directly which is calculated exactly and is more efficient. Alternatively, you can use:

 ```julia
 inner(psi, Apply(H, psi))
 ```

 in which case `Apply(H, psi)` represents the "lazy" evaluation of
 `apply(H, psi)` and internally calls something equivalent to `inner(psi', H, psi)`.

 Although the new behavior seems less convenient, it makes it easier to
 generalize `inner(::MPS, ::MPS)` to other types of inputs, like `MPS` with
 different tag and prime conventions, multiple sites per tensor, `ITensor` inputs, etc.
 """
end

# Implement below, define here so it can be used in `deprecate_make_inds_match!`.
function _log_or_not_dot end

function deprecate_make_inds_match!(
  ::typeof(_log_or_not_dot),
  M1dag::MPST,
  M2::MPST,
  loginner::Bool;
  make_inds_match::Bool=true,
) where {MPST<:AbstractMPS}
  siteindsM1dag = siteinds(all, M1dag)
  siteindsM2 = siteinds(all, M2)
  N = length(M2)
  if any(n -> length(n) > 1, siteindsM1dag) ||
    any(n -> length(n) > 1, siteindsM2) ||
    !hassamenuminds(siteinds, M1dag, M2)
    # If the MPS have more than one site Indices on any site or they don't have
    # the same number of site indices on each site, don't try to make the
    # indices match
    if !hassameinds(siteinds, M1dag, M2)
      n = findfirst(n -> !hassameinds(siteinds(M1dag, n), siteinds(M2, n)), 1:N)
      error(
        """Calling `dot(ϕ::MPS/MPO, ψ::MPS/MPO)` with multiple site indices per
        MPS/MPO tensor but the site indices don't match. Even with `make_inds_match = true`,
        the case of multiple site indices per MPS/MPO is not handled automatically.
        The sites with unmatched site indices are:

            inds(ϕ[$n]) = $(inds(M1dag[n]))

            inds(ψ[$n]) = $(inds(M2[n]))

        Make sure the site indices of your MPO/MPS match. You may need to prime
        one of the MPS, such as `dot(ϕ', ψ)`."""
      )
    end
    make_inds_match = false
  end
  if !hassameinds(siteinds, M1dag, M2) && make_inds_match
    warn_once(inner_mps_mpo_mps_deprecation_warning(), :inner_mps_mps)
    replace_siteinds!(M1dag, siteindsM2)
  end
  return M1dag, M2
end

function _log_or_not_dot(
  M1::MPST, M2::MPST, loginner::Bool; make_inds_match::Bool=true
)::Number where {MPST<:AbstractMPS}
  N = length(M1)
  if length(M2) != N
    throw(DimensionMismatch("inner: mismatched lengths $N and $(length(M2))"))
  end
  M1dag = dag(M1)
  sim!(linkinds, M1dag)
  M1dag, M2 = deprecate_make_inds_match!(
    _log_or_not_dot, M1dag, M2, loginner; make_inds_match
  )
  check_hascommoninds(siteinds, M1dag, M2)
  O = M1dag[1] * M2[1]

  if loginner
    normO = norm(O)
    log_inner_tot = log(normO)
    O ./= normO
  end

  for j in eachindex(M1)[2:end]
    O = (O * M1dag[j]) * M2[j]

    if loginner
      normO = norm(O)
      log_inner_tot += log(normO)
      O ./= normO
    end
  end

  if loginner
    if !isreal(O[]) || real(O[]) < 0
      log_inner_tot += log(complex(O[]))
    end
    return log_inner_tot
  end

  dot_M1_M2 = O[]

  if !isfinite(dot_M1_M2)
    @warn "The inner product (or norm²) you are computing is very large " *
      "($dot_M1_M2). You should consider using `lognorm` or `loginner` instead, " *
      "which will help avoid floating point errors. For example if you are trying " *
      "to normalize your MPS/MPO `A`, the normalized MPS/MPO `B` would be given by " *
      "`B = A ./ z` where `z = exp(lognorm(A) / length(A))`."
  end

  return dot_M1_M2
end

"""
    dot(A::MPS, B::MPS)
    dot(A::MPO, B::MPO)

Same as [`inner`](@ref).

See also [`loginner`](@ref), [`logdot`](@ref).
"""
function dot(M1::MPST, M2::MPST; kwargs...) where {MPST<:AbstractMPS}
  return _log_or_not_dot(M1, M2, false; kwargs...)
end

"""
    logdot(A::MPS, B::MPS)
    logdot(A::MPO, B::MPO)

Same as [`loginner`](@ref).

See also [`inner`](@ref), [`dot`](@ref).
"""
function logdot(M1::MPST, M2::MPST; kwargs...) where {MPST<:AbstractMPS}
  return _log_or_not_dot(M1, M2, true; kwargs...)
end

function make_inds_match_docstring_warning()
  return """
  !!! compat "ITensors 0.3"
    Before ITensors 0.3, `inner` had a keyword argument `make_inds_match` that default to `true`.
    When true, the function attempted to make the site indices match before contracting. So for example, the
    inputs could have different site indices, as long as they have the same dimensions or QN blocks.
    This behavior was fragile since it only worked for MPS with single site indices per tensor,
    and as of ITensors 0.3 has been deprecated. As of ITensors 0.3 you will need to make sure
    the MPS or MPO you input have compatible site indices to contract over, such as by making
    sure the prime levels match properly.
  """
end

"""
    inner(A::MPS, B::MPS)
    inner(A::MPO, B::MPO)

Compute the inner product `⟨A|B⟩`. If `A` and `B` are MPOs, computes the Frobenius inner product.

Use [`loginner`](@ref) to avoid underflow/overflow for taking overlaps of large MPS or MPO.

$(make_inds_match_docstring_warning())

Same as [`dot`](@ref).

See also [`loginner`](@ref), [`logdot`](@ref).
"""
inner(M1::MPST, M2::MPST; kwargs...) where {MPST<:AbstractMPS} = dot(M1, M2; kwargs...)

"""
    loginner(A::MPS, B::MPS)
    loginner(A::MPO, B::MPO)

Compute the logarithm of the inner product `⟨A|B⟩`. If `A` and `B` are MPOs, computes the logarithm of the Frobenius inner product.

This is useful for larger MPS/MPO, where in the limit of large numbers of sites the inner product can diverge or approach zero.

$(make_inds_match_docstring_warning())

Same as [`logdot`](@ref).

See also [`inner`](@ref), [`dot`](@ref).
"""
function loginner(M1::MPST, M2::MPST; kwargs...) where {MPST<:AbstractMPS}
  return logdot(M1, M2; kwargs...)
end

"""
    norm(A::MPS)
    norm(A::MPO)

Compute the norm of the MPS or MPO.

If the MPS or MPO has a well defined orthogonality center, this reduces to the
norm of the orthogonality center tensor. Otherwise, it computes the norm with
the full inner product of the MPS/MPO with itself.

See also [`lognorm`](@ref).
"""
function norm(M::AbstractMPS)
  if isortho(M)
    return norm(M[orthocenter(M)])
  end
  norm2_M = dot(M, M)
  rtol = eps(real(scalartype(M))) * 10
  atol = rtol
  if !IsApprox.isreal(norm2_M, Approx(; rtol=rtol, atol=atol))
    @warn "norm² is $norm2_M, which is not real up to a relative tolerance of " *
      "$rtol and an absolute tolerance of $atol. Taking the real part, which may not be accurate."
  end
  return sqrt(real(norm2_M))
end

"""
    lognorm(A::MPS)
    lognorm(A::MPO)

Compute the logarithm of the norm of the MPS or MPO.

This is useful for larger MPS/MPO that are not gauged, where in the limit of
large numbers of sites the norm can diverge or approach zero.

See also [`norm`](@ref), [`logdot`](@ref).
"""
function lognorm(M::AbstractMPS)
  if isortho(M)
    return log(norm(M[orthocenter(M)]))
  end
  lognorm2_M = logdot(M, M)
  rtol = eps(real(scalartype(M))) * 10
  atol = rtol
  if !IsApprox.isreal(lognorm2_M, Approx(; rtol=rtol, atol=atol))
    @warn "log(norm²) is $lognorm2_M, which is not real up to a relative tolerance " *
      "of $rtol and an absolute tolerance of $atol. Taking the real part, which may not be accurate."
  end
  return 0.5 * real(lognorm2_M)
end

function isapprox(
  x::AbstractMPS,
  y::AbstractMPS;
  atol::Real=0,
  rtol::Real=Base.rtoldefault(
    LinearAlgebra.promote_leaf_eltypes(x), LinearAlgebra.promote_leaf_eltypes(y), atol
  ),
)
  d = norm(x - y)
  if isfinite(d)
    return d <= max(atol, rtol * max(norm(x), norm(y)))
  else
    error("In `isapprox(x::MPS, y::MPS)`, `norm(x - y)` is not finite")
  end
end

# copy an MPS/MPO, but do a deep copy of the tensors in the
# range of the orthogonality center.
function deepcopy_ortho_center(M::AbstractMPS)
  M = copy(M)
  c = ortho_lims(M)
  # TODO: define `getindex(::AbstractMPS, I)` to return `AbstractMPS`
  M[c] = deepcopy(typeof(M)(M[c]))
  return M
end

"""
    normalize(A::MPS; (lognorm!)=[])
    normalize(A::MPO; (lognorm!)=[])

Return a new MPS or MPO `A` that is the same as the original MPS or MPO but with `norm(A) ≈ 1`.

In practice, this evenly spreads `lognorm(A)` over the tensors within the range
of the orthogonality center to avoid numerical overflow in the case of diverging norms.

See also [`normalize!`](@ref), [`norm`](@ref), [`lognorm`](@ref).
"""
function normalize(M::AbstractMPS; (lognorm!)=[])
  return normalize!(deepcopy_ortho_center(M); (lognorm!)=lognorm!)
end

"""
    normalize!(A::MPS; (lognorm!)=[])
    normalize!(A::MPO; (lognorm!)=[])

Change the MPS or MPO `A` in-place such that `norm(A) ≈ 1`. This modifies the
data of the tensors within the orthogonality center.

In practice, this evenly spreads `lognorm(A)` over the tensors within the range
of the orthogonality center to avoid numerical overflow in the case of diverging norms.

If the norm of the input MPS or MPO is 0, normalizing is ill-defined. In this
case, we just return the original MPS or MPO. You can check for this case as follows:

```julia
s = siteinds("S=1/2", 4)
ψ = 0 * randomMPS(s)
lognorm_ψ = []
normalize!(ψ; (lognorm!)=lognorm_ψ)
lognorm_ψ[1] == -Inf # There was an infinite norm
```

See also [`normalize`](@ref), [`norm`](@ref), [`lognorm`](@ref).
"""
function normalize!(M::AbstractMPS; (lognorm!)=[])
  c = ortho_lims(M)
  lognorm_M = lognorm(M)
  push!(lognorm!, lognorm_M)
  if lognorm_M == -Inf
    return M
  end
  z = exp(lognorm_M / length(c))
  # XXX: this is not modifying `M` in-place.
  # M[c] ./= z
  for n in c
    M[n] ./= z
  end
  return M
end

"""
    dist(A::MPS, B::MPS)
    dist(A::MPO, B::MPO)

Compute the Euclidean distance between to MPS/MPO. Equivalent to `norm(A - B)`
but done more efficiently as:

`sqrt(abs(inner(A, A) + inner(B, B) - 2 * real(inner(A, B))))`.

Note that if the MPS/MPO are not normalized, the normalizations may diverge and
  this may not be accurate. For those cases, likely it is best to use `norm(A - B)`
  directly (or `lognorm(A - B)` if you expect the result may be very large).
"""
function dist(A::AbstractMPS, B::AbstractMPS)
  return sqrt(abs(inner(A, A) + inner(B, B) - 2 * real(inner(A, B))))
end

function site_combiners(ψ::AbstractMPS)
  N = length(ψ)
  Cs = Vector{ITensor}(undef, N)
  for n in 1:N
    s = siteinds(all, ψ, n)
    Cs[n] = combiner(s; tags=commontags(s))
  end
  return Cs
end

# The maximum link dimensions when adding MPS/MPO
function _add_maxlinkdims(ψ⃗::AbstractMPS...)
  N = length(ψ⃗[1])
  maxdims = Vector{Int}(undef, N - 1)
  for b in 1:(N - 1)
    maxdims[b] = sum(ψ -> linkdim(ψ, b), ψ⃗)
  end
  return maxdims
end

function +(
  ::Algorithm"densitymatrix", ψ⃗::MPST...; cutoff=1e-15, kwargs...
) where {MPST<:AbstractMPS}
  if !all(ψ -> hassameinds(siteinds, first(ψ⃗), ψ), ψ⃗)
    error("In `+(::MPS/MPO...)`, the input `MPS` or `MPO` do not have the same site
      indices. For example, the site indices of the first site are $(siteinds.(ψ⃗, 1))")
  end

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
    Dₙ, Vₙ, spec = eigen(
      ρₙ;
      ishermitian=true,
      tags=tags(linkind(ψ⃗[1], n - 1)),
      cutoff=cutoff,
      maxdim=add_maxlinkdims[n - 1],
      kwargs...,
    )
    lₙ₋₁ = commonind(Dₙ, Vₙ)

    # Update the total state
    ψ[n] = Vₙ

    # Compute the new density matrix
    C⃗ₙ₋₁ = [ψ⃗[i][n - 1] * C⃗ₙ[i] * dag(Vₙ) for i in 1:Nₘₚₛ]
    C⃗ₙ₋₁′ = [prime(Cₙ₋₁, (s[n - 1], lₙ₋₁)) for Cₙ₋₁ in C⃗ₙ₋₁]
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

function +(::Algorithm"directsum", ψ⃗::MPST...) where {MPST<:AbstractMPS}
  n = length(first(ψ⃗))
  @assert all(ψᵢ -> length(first(ψ⃗)) == length(ψᵢ), ψ⃗)

  # Output tensor
  ϕ = MPST(n)

  # Direct sum first tensor
  j = 1
  l⃗j = map(ψᵢ -> linkind(ψᵢ, j), ψ⃗)
  ϕj, (lj,) = directsum(
    (ψ⃗[i][j] => (l⃗j[i],) for i in 1:length(ψ⃗))...; tags=[tags(first(l⃗j))]
  )
  ljm_prev = lj
  ϕ[j] = ϕj
  for j in 2:(n - 1)
    l⃗jm = map(ψᵢ -> linkind(ψᵢ, j - 1), ψ⃗)
    l⃗j = map(ψᵢ -> linkind(ψᵢ, j), ψ⃗)
    ϕj, (ljm, lj) = directsum(
      (ψ⃗[i][j] => (l⃗jm[i], l⃗j[i]) for i in 1:length(ψ⃗))...;
      tags=[tags(first(l⃗jm)), tags(first(l⃗j))],
    )
    ϕj = replaceind(ϕj, ljm => dag(ljm_prev))
    ljm_prev = lj
    ϕ[j] = ϕj
  end
  j = n
  l⃗jm = map(ψᵢ -> linkind(ψᵢ, j - 1), ψ⃗)
  ϕj, (ljm,) = directsum(
    (ψ⃗[i][j] => (l⃗jm[i],) for i in 1:length(ψ⃗))...; tags=[tags(first(l⃗jm))]
  )
  ϕj = replaceind(ϕj, ljm => dag(ljm_prev))
  ϕ[j] = ϕj
  return ϕ
end

"""
    +(A::MPS/MPO...; kwargs...)
    add(A::MPS/MPO...; kwargs...)

Add arbitrary numbers of MPS/MPO with each other, optionally truncating the results.

A cutoff of 1e-15 is used by default, and in general users should set their own
cutoff for their particular application.

# Keywords

- `cutoff::Real`: singular value truncation cutoff
- `maxdim::Int`: maximum MPS/MPO bond dimension
- `alg = "densitymatrix"`: `"densitymatrix"` or `"directsum"`. `"densitymatrix"` adds the MPS/MPO
   by adding up and diagoanlizing local density matrices site by site in a single
   sweep through the system, truncating the density matrix with `cutoff` and `maxdim`.
   `"directsum"` performs a direct sum of each tensors on each site of the input
   MPS/MPO being summed. It doesn't perform any truncation, and therefore ignores
   `cutoff` and `maxdim`. The bond dimension of the output is the sum of the bond
   dimensions of the inputs. You can truncate the resulting MPS/MPO with the `truncate!` function.

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
function +(ψ⃗::AbstractMPS...; alg=Algorithm"densitymatrix"(), kwargs...)
  return +(Algorithm(alg), ψ⃗...; kwargs...)
end

+(ψ::AbstractMPS) = ψ

add(ψ⃗::AbstractMPS...; kwargs...) = +(ψ⃗...; kwargs...)

-(ψ₁::AbstractMPS, ψ₂::AbstractMPS; kwargs...) = +(ψ₁, -ψ₂; kwargs...)

add(A::T, B::T; kwargs...) where {T<:AbstractMPS} = +(A, B; kwargs...)

"""
    sum(A::Vector{MPS}; kwargs...)

    sum(A::Vector{MPO}; kwargs...)

Add multiple MPS/MPO with each other, with some optional
truncation.

# Keywords

- `cutoff::Real`: singular value truncation cutoff
- `maxdim::Int`: maximum MPS/MPO bond dimension
"""
function sum(ψ⃗::Vector{T}; kwargs...) where {T<:AbstractMPS}
  iszero(length(ψ⃗)) && return T()
  isone(length(ψ⃗)) && return copy(only(ψ⃗))
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
function orthogonalize!(M::AbstractMPS, j::Int; kwargs...)
  @debug_check begin
    if !(1 <= j <= length(M))
      error("Input j=$j to `orthogonalize!` out of range (valid range = 1:$(length(M)))")
    end
  end
  while leftlim(M) < (j - 1)
    (leftlim(M) < 0) && setleftlim!(M, 0)
    b = leftlim(M) + 1
    linds = uniqueinds(M[b], M[b + 1])
    lb = linkind(M, b)
    if !isnothing(lb)
      ltags = tags(lb)
    else
      ltags = TagSet("Link,l=$b")
    end
    L, R = factorize(M[b], linds; tags=ltags, kwargs...)
    M[b] = L
    M[b + 1] *= R
    setleftlim!(M, b)
    if rightlim(M) < leftlim(M) + 2
      setrightlim!(M, leftlim(M) + 2)
    end
  end

  N = length(M)

  while rightlim(M) > (j + 1)
    (rightlim(M) > (N + 1)) && setrightlim!(M, N + 1)
    b = rightlim(M) - 2
    rinds = uniqueinds(M[b + 1], M[b])
    lb = linkind(M, b)
    if !isnothing(lb)
      ltags = tags(lb)
    else
      ltags = TagSet("Link,l=$b")
    end
    L, R = factorize(M[b + 1], rinds; tags=ltags, kwargs...)
    M[b + 1] = L
    M[b] *= R

    setrightlim!(M, b + 1)
    if leftlim(M) > rightlim(M) - 2
      setleftlim!(M, rightlim(M) - 2)
    end
  end
  return M
end

# Allows overloading `orthogonalize!` based on the projected
# MPO type. By default just calls `orthogonalize!` on the MPS.
function orthogonalize!(PH, M::AbstractMPS, j::Int; kwargs...)
  return orthogonalize!(M, j; kwargs...)
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

Keyword arguments:
* `site_range`=1:N - only truncate the MPS bonds between these sites
"""
function truncate!(M::AbstractMPS; alg="frobenius", kwargs...)
  return truncate!(Algorithm(alg), M; kwargs...)
end

function truncate!(
  ::Algorithm"frobenius", M::AbstractMPS; site_range=1:length(M), kwargs...
)
  N = length(M)

  # Left-orthogonalize all tensors to make
  # truncations controlled
  orthogonalize!(M, last(site_range))

  # Perform truncations in a right-to-left sweep
  for j in reverse((first(site_range) + 1):last(site_range))
    rinds = uniqueinds(M[j], M[j - 1])
    ltags = tags(commonind(M[j], M[j - 1]))
    U, S, V = svd(M[j], rinds; lefttags=ltags, kwargs...)
    M[j] = U
    M[j - 1] *= (S * V)
    setrightlim!(M, j)
  end
  return M
end

function truncate(ψ0::AbstractMPS; kwargs...)
  ψ = copy(ψ0)
  truncate!(ψ; kwargs...)
  return ψ
end

# Make `*` an alias for `contract` of two `AbstractMPS`
*(A::AbstractMPS, B::AbstractMPS; kwargs...) = contract(A, B; kwargs...)

function _apply_to_orthocenter!(f, ψ::AbstractMPS, x)
  limsψ = ortho_lims(ψ)
  n = first(limsψ)
  ψ[n] = f(ψ[n], x)
  return ψ
end

function _apply_to_orthocenter(f, ψ::AbstractMPS, x)
  return _apply_to_orthocenter!(f, copy(ψ), x)
end

"""
    ψ::MPS/MPO * α::Number

Scales the MPS or MPO by the provided number.

Currently, this works by scaling one of the sites within the orthogonality limits.
"""
(ψ::AbstractMPS * α::Number) = _apply_to_orthocenter(*, ψ, α)

"""
    α::Number * ψ::MPS/MPO

Scales the MPS or MPO by the provided number.

Currently, this works by scaling one of the sites within the orthogonality limits.
"""
(α::Number * ψ::AbstractMPS) = ψ * α

(ψ::AbstractMPS / α::Number) = _apply_to_orthocenter(/, ψ, α)

-(ψ::AbstractMPS) = -1 * ψ

LinearAlgebra.rmul!(ψ::AbstractMPS, α::Number) = _apply_to_orthocenter!(*, ψ, α)

"""
    setindex!(::Union{MPS, MPO}, ::Union{MPS, MPO},
              r::UnitRange{Int64})

Sets a contiguous range of MPS/MPO tensors
"""
function setindex!(ψ::MPST, ϕ::MPST, r::UnitRange{Int64}) where {MPST<:AbstractMPS}
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
  T = ITensor(QN(), s1', s2', dag(s1), dag(s2))
  for b in nzblocks(T)
    dval = 1.0
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
    setindex!(ψ::Union{MPS, MPO}, A::ITensor, r::UnitRange{Int};
              orthocenter::Int = last(r), perm = nothing, kwargs...)
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
function setindex!(
  ψ::MPST,
  A::ITensor,
  r::UnitRange{Int};
  orthocenter::Integer=last(r),
  perm=nothing,
  kwargs...,
) where {MPST<:AbstractMPS}
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
  lind = linkind(ψ, firstsite - 1)
  rind = linkind(ψ, lastsite)

  sites = [siteinds(ψ, j) for j in firstsite:lastsite]

  #s = collect(Iterators.flatten(sites))
  indsA = filter(x -> !isnothing(x), [lind, Iterators.flatten(sites)..., rind])
  @assert hassameinds(A, indsA)

  # For MPO case, restrict to 0 prime level
  #sites = filter(hasplev(0), sites)

  if !isnothing(perm)
    sites0 = sites
    sites = sites0[[perm...]]
    # Check if the site indices
    # are fermionic
    if !using_auto_fermion() && any(anyfermionic, sites)
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

  ψA = MPST(A, sites; leftinds=lind, orthocenter=orthocenter - first(r) + 1, kwargs...)
  #@assert prod(ψA) ≈ A

  ψ[firstsite:lastsite] = ψA

  return ψ
end

function setindex!(
  ψ::MPST, A::ITensor, r::UnitRange{Int}, args::Pair{Symbol}...; kwargs...
) where {MPST<:AbstractMPS}
  return setindex!(ψ, A, r; args..., kwargs...)
end

replacesites!(ψ::AbstractMPS, args...; kwargs...) = setindex!(ψ, args...; kwargs...)

replacesites(ψ::AbstractMPS, args...; kwargs...) = setindex!(copy(ψ), args...; kwargs...)

_number_inds(s::Index) = 1
_number_inds(s::IndexSet) = length(s)
_number_inds(sites) = sum(_number_inds(s) for s in sites)

"""
    MPS(A::ITensor, sites; kwargs...)
    MPO(A::ITensor, sites; kwargs...)

Construct an MPS/MPO from an ITensor `A` by decomposing it site
by site according to the site indices `sites`.

# Keywords

- `leftinds = nothing`: optional left dangling indices. Indices that are not
   in `sites` and `leftinds` will be dangling off of the right side of the MPS/MPO.
- `orthocenter::Integer = length(sites)`: the desired final orthogonality
   center of the output MPS/MPO.
- `cutoff`: the desired truncation error at each link.
- `maxdim`: the maximum link dimension.
"""
function (::Type{MPST})(
  A::ITensor, sites; leftinds=nothing, orthocenter::Integer=length(sites), kwargs...
) where {MPST<:AbstractMPS}
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
  for n in 1:(N - 1)
    Lis = IndexSet(sites[n])
    if !isnothing(l)
      Lis = unioninds(Lis, l)
    end
    L, R = factorize(Ã, Lis; kwargs..., tags="Link,n=$n", ortho="left")
    l = commonind(L, R)
    ψ[n] = L
    Ã = R
  end
  ψ[N] = Ã
  M = MPST(ψ)
  setleftlim!(M, N - 1)
  setrightlim!(M, N + 1)
  orthogonalize!(M, orthocenter)
  return M
end

function (::Type{MPST})(A::AbstractArray, sites; kwargs...) where {MPST<:AbstractMPS}
  return MPST(itensor(A, sites...), sites; kwargs...)
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
  ψ[b:(b + 1), orthocenter=orthocenter, perm=[2, 1], kwargs...] = ψ[b] * ψ[b + 1]
  return ψ
end

"""
    movesite(::Union{MPS, MPO}, n1n2::Pair{Int, Int})

Create a new MPS/MPO where the site at `n1` is moved to `n2`,
for a pair `n1n2 = n1 => n2`.

This is done with a series a pairwise swaps, and can introduce
a lot of entanglement into your state, so use with caution.
"""
function movesite(
  ψ::AbstractMPS, n1n2::Pair{Int,Int}; orthocenter::Integer=last(n1n2), kwargs...
)
  n1, n2 = n1n2
  n1 == n2 && return copy(ψ)
  ψ = orthogonalize(ψ, n2)
  r = n1:(n2 - 1)
  ortho = "left"
  if n1 > n2
    r = reverse(n2:(n1 - 1))
    ortho = "right"
  end
  for n in r
    ψ = swapbondsites(ψ, n; ortho=ortho, kwargs...)
  end
  ψ = orthogonalize(ψ, orthocenter)
  return ψ
end

# Helper function for permuting a vector for the
# movesites function.
function _movesite(ns::Vector{Int}, n1n2::Pair{Int,Int})
  n1, n2 = n1n2
  n1 == n2 && return copy(ns)
  r = n1:(n2 - 1)
  if n1 > n2
    r = reverse(n2:(n1 - 1))
  end
  for n in r
    ns = replace(ns, n => n + 1, n + 1 => n)
  end
  return ns
end

function _movesites(ψ::AbstractMPS, ns::Vector{Int}, ns′::Vector{Int}; kwargs...)
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
function movesites(ψ::AbstractMPS, nsns′::Vector{Pair{Int,Int}}; kwargs...)
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
function movesites(ψ::AbstractMPS, ns, ns′; kwargs...)
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
    apply(o::ITensor, ψ::Union{MPS, MPO}, [ns::Vector{Int}]; kwargs...)
    product([...])

Get the product of the operator `o` with the MPS/MPO `ψ`,
where the operator is applied to the sites `ns`. If `ns`
are not specified, the sites are determined by the common indices
between `o` and the site indices of `ψ`.

If `ns` are non-contiguous, the sites of the MPS are
moved to be contiguous. By default, the sites are moved
back to their original locations. You can leave them where
they are by setting the keyword argument `move_sites_back`
to false.

# Keywords

- `cutoff::Real`: singular value truncation cutoff.
- `maxdim::Int`: maximum MPS/MPO dimension.
- `apply_dag::Bool = false`: apply the gate and the dagger of the gate (only
   relevant for MPO evolution).
- `move_sites_back::Bool = true`: after the ITensors are applied to the MPS or
   MPO, move the sites of the MPS or MPO back to their original locations.
"""
function product(
  o::ITensor,
  ψ::AbstractMPS,
  ns=findsites(ψ, o);
  move_sites_back::Bool=true,
  apply_dag::Bool=false,
  kwargs...,
)
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
  ϕ = product(o, ϕ; apply_dag=apply_dag)
  ψ[ns′[1]:ns′[end], kwargs...] = ϕ
  if move_sites_back
    # Move the sites back to their original positions
    ψ = movesites(ψ, ns′ .=> ns; kwargs...)
  end
  return ψ
end

"""
    apply(As::Vector{<:ITensor}, M::Union{MPS, MPO}; kwargs...)
    product([...])

Apply the ITensors `As` to the MPS or MPO `M`, treating them as gates or
matrices from pairs of prime or unprimed indices.

# Keywords

- `cutoff::Real`: singular value truncation cutoff.
- `maxdim::Int`: maximum MPS/MPO dimension.
- `apply_dag::Bool = false`: apply the gate and the dagger of the gate
  (only relevant for MPO evolution).
- `move_sites_back::Bool = true`: after the ITensor is applied to the MPS or
   MPO, move the sites of the MPS or MPO back to their original locations.

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
ψ0 = MPS(s, "↑")

# Apply the gates.
ψ = apply(gates, ψ0; cutoff = 1e-15)

# Test against exact (full) wavefunction
prodψ = apply(gates, prod(ψ0))
@show prod(ψ) ≈ prodψ

# The result is:
# σz₃ σz₂ σz₁ σx₃ σx₂ σx₁ |↑↑↑⟩ = -|↓↓↓⟩
@show inner(ψ, MPS(s, "↓")) == -1
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
ψ0 = MPS(s, n -> n == 1 ? "↓" : "↑")

# The result is:
# σz₃ CX₁₃ |↓↑↑⟩ = -|↓↑↓⟩
ψ = apply(ops(os, s), ψ0; cutoff = 1e-15)
@show inner(ψ, MPS(s, n -> n == 1 || n == 3 ? "↓" : "↑")) == -1
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
ψ0 = MPS(s, n -> n == 1 ? "↓" : "↑")
expτH = ops(os, s)
ψτ = apply(expτH, ψ0)
```
"""
function product(
  As::Vector{ITensor},
  ψ::AbstractMPS;
  move_sites_back_between_gates::Bool=true,
  move_sites_back::Bool=true,
  kwargs...,
)
  Aψ = ψ
  for A in As
    Aψ = product(A, Aψ; move_sites_back=move_sites_back_between_gates, kwargs...)
  end
  if !move_sites_back_between_gates && move_sites_back
    s = siteinds(Aψ)
    ns = 1:length(ψ)
    ñs = [findsite(ψ, i) for i in s]
    Aψ = movesites(Aψ, ns .=> ñs; kwargs...)
  end
  return Aψ
end

# Apply in the reverse order for proper order of operations
# For example:
#
# s = siteinds("Qubit", 1)
# ψ = randomMPS(s)
#
# # U = Z₁X₁
# U = Prod{Op}()
# U = ("X", 1) * U
# U = ("Z", 1) * U
#
# # U|ψ⟩ = Z₁X₁|ψ⟩
# apply(U,
function product(o::Prod{ITensor}, ψ::AbstractMPS; kwargs...)
  return product(reverse(terms(o)), ψ; kwargs...)
end

function (o::Prod{ITensor})(ψ::AbstractMPS; kwargs...)
  return apply(o, ψ; kwargs...)
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
hasqns(M::AbstractMPS) = any(hasqns, data(M))

# Trait type version of hasqns
# Note this is not inferrable, so hasqns would be preferred
symmetrystyle(M::AbstractMPS) = symmetrystyle(data(M))

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
  for j in (M.llim + 1):(M.rlim - 1)
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

Split the QN blocks of the links of the MPS or MPO into dimension 1 blocks.
Then, only keep the blocks with `norm(b) > tol`.

This can make the ITensors of the MPS/MPO more sparse, and is particularly
helpful as a preprocessing step on a local Hamiltonian MPO for DMRG.
"""
function splitblocks!(::typeof(linkinds), M::AbstractMPS; tol=0)
  for i in eachindex(M)[1:(end - 1)]
    l = linkind(M, i)
    if !isnothing(l)
      @preserve_ortho M begin
        M[i] = splitblocks(M[i], l)
        M[i + 1] = splitblocks(M[i + 1], l)
      end
    end
  end
  return M
end

function splitblocks(::typeof(linkinds), M::AbstractMPS; tol=0)
  return splitblocks!(linkinds, copy(M); tol=0)
end

removeqns(M::AbstractMPS) = map(removeqns, M; set_limits=false)
function removeqn(M::AbstractMPS, qn_name::String)
  return map(m -> removeqn(m, qn_name), M; set_limits=false)
end

#
# Broadcasting
#

BroadcastStyle(MPST::Type{<:AbstractMPS}) = Style{MPST}()
function BroadcastStyle(::Style{MPST}, ::DefaultArrayStyle{N}) where {N,MPST<:AbstractMPS}
  return Style{MPST}()
end

broadcastable(ψ::AbstractMPS) = ψ
function copyto!(ψ::AbstractMPS, b::Broadcasted)
  copyto!(data(ψ), b)
  # In general, we assume the broadcast operation
  # will mess up the orthogonality
  # TODO: special case for `prime`, `settags`, etc.
  reset_ortho_lims!(ψ)
  return ψ
end

function similar(bc::Broadcasted{Style{MPST}}, ElType::Type) where {MPST<:AbstractMPS}
  return similar(Array{ElType}, axes(bc))
end

function similar(bc::Broadcasted{Style{MPST}}, ::Type{ITensor}) where {MPST<:AbstractMPS}
  # In general, we assume the broadcast operation
  # will mess up the orthogonality so we use
  # a generic constructor where we don't specify
  # the orthogonality limits.
  return MPST(similar(Array{ITensor}, axes(bc)))
end

#
# Printing functions
#

function Base.show(io::IO, M::AbstractMPS)
  print(io, "$(typeof(M))")
  (length(M) > 0) && print(io, "\n")
  for i in eachindex(M)
    if !isassigned(M, i)
      println(io, "#undef")
    else
      A = M[i]
      if order(A) != 0
        println(io, "[$i] $(inds(A))")
      else
        println(io, "[$i] ITensor()")
      end
    end
  end
end
