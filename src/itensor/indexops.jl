#########################
# ITensor Index Functions
#

"""
    inds(T::ITensor)

Return the indices of the ITensor as a Tuple.
"""
inds(T::ITensor) = inds(tensor(T))

"""
    ind(T::ITensor, i::Int)

Get the Index of the ITensor along dimension i.
"""
ind(T::ITensor, i::Int) = ind(tensor(T), i)

"""
    eachindex(A::ITensor)

Create an iterable object for visiting each element of the ITensor `A` (including structually
zero elements for sparse tensors).

For example, for dense tensors this may return `1:length(A)`, while for sparse tensors
it may return a Cartesian range.
"""
eachindex(A::ITensor) = eachindex(tensor(A))

# TODO: name this `inds` or `indscollection`?
itensor2inds(A::ITensor)::Any = inds(A)
itensor2inds(A::Tensor) = inds(A)
itensor2inds(i::Index) = (i,)
itensor2inds(A) = A
function map_itensor2inds(A::Tuple{Vararg{Any,N}}) where {N}
  return ntuple(i -> itensor2inds(A[i]), Val(N))
end

# in
hasind(A, i::Index) = i ∈ itensor2inds(A)

# issubset
hasinds(A, is) = is ⊆ itensor2inds(A)
hasinds(A, is::Index...) = hasinds(A, is)

"""
    hasinds(is...)

Returns an anonymous function `x -> hasinds(x, is...)` which
accepts an ITensor or IndexSet and returns `true` if the
ITensor or IndexSet has the indices `is`.
"""
hasinds(is::Indices) = x -> hasinds(x, is)
hasinds(is::Index...) = hasinds(is)

"""
    hascommoninds(A, B; kwargs...)

    hascommoninds(B; kwargs...) -> f::Function

Check if the ITensors or sets of indices `A` and `B` have
common indices.

If only one ITensor or set of indices `B` is passed, return a
function `f` such that `f(A) = hascommoninds(A, B; kwargs...)`
"""
hascommoninds(A, B; kwargs...) = !isnothing(commonind(A, B; kwargs...))

hascommoninds(B; kwargs...) = x -> hascommoninds(x, B; kwargs...)

# issetequal
hassameinds(A, B) = issetequal(itensor2inds(A), itensor2inds(B))

# Apply the Index set function and then filter the results
function filter_inds_set_function(
  ffilter::Function, fset::Function, A::Vararg{Any,N}
) where {N}
  return filter(ffilter, fset(map_itensor2inds(A)...))
end

function filter_inds_set_function(fset::Function, A...; kwargs...)
  return filter_inds_set_function(fmatch(; kwargs...), fset, A...)
end

for (finds, fset) in (
  (:commoninds, :_intersect),
  (:noncommoninds, :_symdiff),
  (:uniqueinds, :_setdiff),
  (:unioninds, :_union),
)
  @eval begin
    $finds(args...; kwargs...) = filter_inds_set_function($fset, args...; kwargs...)
  end
end

for find in (:commonind, :noncommonind, :uniqueind, :unionind)
  @eval begin
    $find(args...; kwargs...) = getfirst($(Symbol(find, :s))(args...; kwargs...))
  end
end

# intersect
@doc """
    commoninds(A, B; kwargs...)

Return a Vector with indices that are common between the indices of `A` and `B`
(the set intersection, similar to `Base.intersect`).
""" commoninds

# firstintersect
@doc """
    commonind(A, B; kwargs...)

Return the first `Index` common between the indices of `A` and `B`.

See also [`commoninds`](@ref).
""" commonind

# symdiff
@doc """
    noncommoninds(A, B; kwargs...)

Return a Vector with indices that are not common between the indices of `A` and
`B` (the symmetric set difference, similar to `Base.symdiff`).
""" noncommoninds

# firstsymdiff
@doc """
    noncommonind(A, B; kwargs...)

Return the first `Index` not common between the indices of `A` and `B`.

See also [`noncommoninds`](@ref).
""" noncommonind

# setdiff
@doc """
    uniqueinds(A, B; kwargs...)

Return Vector with indices that are unique to the set of indices of `A` and not
in `B` (the set difference, similar to `Base.setdiff`).
""" uniqueinds

# firstsetdiff
@doc """
    uniqueind(A, B; kwargs...)

Return the first `Index` unique to the set of indices of `A` and not in `B`.

See also [`uniqueinds`](@ref).
""" uniqueind

# union
@doc """
    unioninds(A, B; kwargs...)

Return a Vector with indices that are the union of the indices of `A` and `B`
(the set union, similar to `Base.union`).
""" unioninds

# firstunion
@doc """
    unionind(A, B; kwargs...)

Return the first `Index` in the union of the indices of `A` and `B`.

See also [`unioninds`](@ref).
""" unionind

firstind(A...; kwargs...) = getfirst(map_itensor2inds(A)...; kwargs...)

filterinds(f::Function, A...) = filter(f, map_itensor2inds(A)...)
filterinds(A...; kwargs...) = filter(map_itensor2inds(A)...; kwargs...)

# Faster version when no filtering is requested
filterinds(A::ITensor) = inds(A)
filterinds(is::Indices) = is

# For backwards compatibility
inds(A...; kwargs...) = filterinds(A...; kwargs...)

# in-place versions of priming and tagging
for fname in (
  :prime,
  :setprime,
  :noprime,
  :replaceprime,
  :swapprime,
  :addtags,
  :removetags,
  :replacetags,
  :settags,
  :swaptags,
  :replaceind,
  :replaceinds,
  :swapind,
  :swapinds,
)
  @eval begin
    $fname(f::Function, A::ITensor, args...) = ITensor($fname(f, tensor(A), args...))

    # Inlining makes the ITensor functions slower
    @noinline function $fname(f::Function, A::Tensor, args...)
      return setinds(A, $fname(f, inds(A), args...))
    end

    function $(Symbol(fname, :!))(f::Function, A::ITensor, args...)
      return settensor!(A, $fname(f, tensor(A), args...))
    end

    $fname(A::ITensor, args...; kwargs...) = itensor($fname(tensor(A), args...; kwargs...))

    # Inlining makes the ITensor functions slower
    @noinline function $fname(A::Tensor, args...; kwargs...)
      return setinds(A, $fname(inds(A), args...; kwargs...))
    end

    function $(Symbol(fname, :!))(A::ITensor, args...; kwargs...)
      return settensor!(A, $fname(tensor(A), args...; kwargs...))
    end
  end
end

priming_tagging_doc = """
Optionally, only modify the indices with the specified keyword arguments.

# Arguments
- `tags = nothing`: if specified, only modify Index `i` if `hastags(i, tags) == true`.
- `plev = nothing`: if specified, only modify Index `i` if `hasplev(i, plev) == true`.

The ITensor functions come in two versions, `f` and `f!`. The latter modifies
the ITensor in-place. In both versions, the ITensor storage is not modified or
copied (so it returns an ITensor with a view of the original storage).
"""

@doc """
    prime[!](A::ITensor, plinc::Int = 1; <keyword arguments>) -> ITensor

    prime(inds, plinc::Int = 1; <keyword arguments>) -> IndexSet

Increase the prime level of the indices of an ITensor or collection of indices.

$priming_tagging_doc
""" prime(::ITensor, ::Any...)

@doc """
    setprime[!](A::ITensor, plev::Int; <keyword arguments>) -> ITensor

    setprime(inds, plev::Int; <keyword arguments>) -> IndexSet

Set the prime level of the indices of an ITensor or collection of indices.

$priming_tagging_doc
""" setprime(::ITensor, ::Any...)

@doc """
    noprime[!](A::ITensor; <keyword arguments>) -> ITensor

    noprime(inds; <keyword arguments>) -> IndexSet

Set the prime level of the indices of an ITensor or collection of indices to zero.

$priming_tagging_doc
""" noprime(::ITensor, ::Any...)

@doc """
    replaceprime[!](A::ITensor, plold::Int, plnew::Int; <keyword arguments>) -> ITensor
    replaceprime[!](A::ITensor, plold => plnew; <keyword arguments>) -> ITensor
    mapprime[!](A::ITensor, <arguments>; <keyword arguments>) -> ITensor

    replaceprime(inds, plold::Int, plnew::Int; <keyword arguments>)
    replaceprime(inds::IndexSet, plold => plnew; <keyword arguments>)
    mapprime(inds, <arguments>; <keyword arguments>)

Set the prime level of the indices of an ITensor or collection of indices with
prime level `plold` to `plnew`.

$priming_tagging_doc
""" mapprime(::ITensor, ::Any...)

@doc """
    swapprime[!](A::ITensor, pl1::Int, pl2::Int; <keyword arguments>) -> ITensor
    swapprime[!](A::ITensor, pl1 => pl2; <keyword arguments>) -> ITensor

    swapprime(inds, pl1::Int, pl2::Int; <keyword arguments>)
    swapprime(inds, pl1 => pl2; <keyword arguments>)

Set the prime level of the indices of an ITensor or collection of indices with
prime level `pl1` to `pl2`, and those with prime level `pl2` to `pl1`.

$priming_tagging_doc
""" swapprime(::ITensor, ::Any...)

@doc """
    addtags[!](A::ITensor, ts::String; <keyword arguments>) -> ITensor

    addtags(inds, ts::String; <keyword arguments>)

Add the tags `ts` to the indices of an ITensor or collection of indices.

$priming_tagging_doc
""" addtags(::ITensor, ::Any...)

@doc """
    removetags[!](A::ITensor, ts::String; <keyword arguments>) -> ITensor

    removetags(inds, ts::String; <keyword arguments>)

Remove the tags `ts` from the indices of an ITensor or collection of indices.

$priming_tagging_doc
""" removetags(::ITensor, ::Any...)

@doc """
    settags[!](A::ITensor, ts::String; <keyword arguments>) -> ITensor

    settags(is::IndexSet, ts::String; <keyword arguments>) -> IndexSet

Set the tags of the indices of an ITensor or IndexSet to `ts`.

$priming_tagging_doc
""" settags(::ITensor, ::Any...)

@doc """
    replacetags[!](A::ITensor, tsold::String, tsnew::String; <keyword arguments>) -> ITensor

    replacetags(is::IndexSet, tsold::String, tsnew::String; <keyword arguments>) -> IndexSet

Replace the tags `tsold` with `tsnew` for the indices of an ITensor.

$priming_tagging_doc
""" replacetags(::ITensor, ::Any...)

@doc """
    swaptags[!](A::ITensor, ts1::String, ts2::String; <keyword arguments>) -> ITensor

    swaptags(is::IndexSet, ts1::String, ts2::String; <keyword arguments>) -> IndexSet

Swap the tags `ts1` with `ts2` for the indices of an ITensor.

$priming_tagging_doc
""" swaptags(::ITensor, ::Any...)

@doc """
    replaceind[!](A::ITensor, i1::Index, i2::Index) -> ITensor

Replace the Index `i1` with the Index `i2` in the ITensor.

The indices must have the same space (i.e. the same dimension and QNs, if applicable).
""" replaceind(::ITensor, ::Any...)

@doc """
    replaceinds(A::ITensor, inds1, inds2) -> ITensor

    replaceinds!(A::ITensor, inds1, inds2)

Replace the Index `inds1[n]` with the Index `inds2[n]` in the ITensor, where `n`
runs from `1` to `length(inds1) == length(inds2)`.

The indices must have the same space (i.e. the same dimension and QNs, if applicable).

The storage of the ITensor is not modified or copied (the output ITensor is a
view of the input ITensor).
""" replaceinds(::ITensor, ::Any...)

@doc """
    swapind(A::ITensor, i1::Index, i2::Index) -> ITensor

    swapind!(A::ITensor, i1::Index, i2::Index)

Swap the Index `i1` with the Index `i2` in the ITensor.

The indices must have the same space (i.e. the same dimension and QNs, if applicable).
""" swapind(::ITensor, ::Any...)

@doc """
    swapinds(A::ITensor, inds1, inds2) -> ITensor

    swapinds!(A::ITensor, inds1, inds2)

Swap the Index `inds1[n]` with the Index `inds2[n]` in the ITensor, where `n`
runs from `1` to `length(inds1) == length(inds2)`.

The indices must have the same space (i.e. the same dimension and QNs, if applicable).

The storage of the ITensor is not modified or copied (the output ITensor is a
view of the input ITensor).
""" swapinds(::ITensor, ::Any...)

# XXX: rename to:
# hastags(any, A, ts)
"""
    anyhastags(A::ITensor, ts::Union{String, TagSet})
    hastags(A::ITensor, ts::Union{String, TagSet})

Check if any of the indices in the ITensor have the specified tags.
"""
anyhastags(A::ITensor, ts) = anyhastags(inds(A), ts)

hastags(A::ITensor, ts) = hastags(inds(A), ts)

# XXX: rename to:
# hastags(all, A, ts)
"""
    allhastags(A::ITensor, ts::Union{String, TagSet})

Check if all of the indices in the ITensor have the specified tags.
"""
allhastags(A::ITensor, ts) = allhastags(inds(A), ts)

# Returns a tuple of pairs of indices, where the pairs
# are determined by the prime level pairs `plev` and
# tag pairs `tags`.
function indpairs(T::ITensor; plev::Pair{Int,Int}=0 => 1, tags::Pair=ts"" => ts"")
  is1 = filterinds(T; plev=first(plev), tags=first(tags))
  is2 = filterinds(T; plev=last(plev), tags=last(tags))
  is2to1 = replacetags(mapprime(is2, last(plev) => first(plev)), last(tags) => first(tags))
  is_first = commoninds(is1, is2to1)
  is_last = replacetags(
    mapprime(is_first, first(plev) => last(plev)), first(tags) => last(tags)
  )
  is_last = permute(commoninds(T, is_last), is_last)
  return is_first .=> is_last
end

#########################
# End ITensor Index Functions
#