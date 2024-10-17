# TODO: add
# splice[!]
# union[!] (∪)
# intersect[!] (∩)
# setdiff[!]
# symdiff[!]
# unique[!]

# unionsorted[!]
# setdiffsorted[!]
# deletesorted[!] (delete all or one?)
# deletesortedfirst[!] (delete all or one?)

Base.resize!(vec::AbstractSmallVector, len) = throw(NotImplemented())

@inline function resize(vec::AbstractSmallVector, len)
  mvec = thaw(vec)
  resize!(mvec, len)
  return convert(similar_type(vec), mvec)
end

@inline function Base.empty!(vec::AbstractSmallVector)
  resize!(vec, 0)
  return vec
end

@inline function empty(vec::AbstractSmallVector)
  mvec = thaw(vec)
  empty!(mvec)
  return convert(similar_type(vec), mvec)
end

@inline function StaticArrays.setindex(vec::AbstractSmallVector, item, index::Integer)
  @boundscheck checkbounds(vec, index)
  mvec = thaw(vec)
  @inbounds mvec[index] = item
  return convert(similar_type(vec), mvec)
end

@inline function Base.push!(vec::AbstractSmallVector, item)
  resize!(vec, length(vec) + 1)
  @inbounds vec[length(vec)] = item
  return vec
end

@inline function StaticArrays.push(vec::AbstractSmallVector, item)
  mvec = thaw(vec)
  push!(mvec, item)
  return convert(similar_type(vec), mvec)
end

@inline function Base.pop!(vec::AbstractSmallVector)
  resize!(vec, length(vec) - 1)
  return vec
end

@inline function StaticArrays.pop(vec::AbstractSmallVector)
  mvec = thaw(vec)
  pop!(mvec)
  return convert(similar_type(vec), mvec)
end

@inline function Base.pushfirst!(vec::AbstractSmallVector, item)
  insert!(vec, firstindex(vec), item)
  return vec
end

# Don't `@inline`, makes it slower.
function StaticArrays.pushfirst(vec::AbstractSmallVector, item)
  mvec = thaw(vec)
  pushfirst!(mvec, item)
  return convert(similar_type(vec), mvec)
end

@inline function Base.popfirst!(vec::AbstractSmallVector)
  circshift!(vec, -1)
  resize!(vec, length(vec) - 1)
  return vec
end

# Don't `@inline`, makes it slower.
function StaticArrays.popfirst(vec::AbstractSmallVector)
  mvec = thaw(vec)
  popfirst!(mvec)
  return convert(similar_type(vec), mvec)
end

# This implementation of `midpoint` is performance-optimized but safe
# only if `lo <= hi`.
# TODO: Replace with `Base.midpoint`.
midpoint(lo::T, hi::T) where {T<:Integer} = lo + ((hi - lo) >>> 0x01)
midpoint(lo::Integer, hi::Integer) = midpoint(promote(lo, hi)...)

@inline function Base.reverse!(vec::AbstractSmallVector)
  start, stop = firstindex(vec), lastindex(vec)
  r = stop
  @inbounds for i in start:midpoint(start, stop-1)
    vec[i], vec[r] = vec[r], vec[i]
    r -= 1
  end
  return vec
end

@inline function Base.reverse!(
  vec::AbstractSmallVector, start::Integer, stop::Integer=lastindex(v)
)
  reverse!(smallview(vec, start, stop))
  return vec
end

@inline function Base.circshift!(vec::AbstractSmallVector, shift::Integer)
  start, stop = firstindex(vec), lastindex(vec)
  n = length(vec)
  n == 0 && return vec
  shift = mod(shift, n)
  shift == 0 && return vec
  reverse!(smallview(vec, start, stop - shift))
  reverse!(smallview(vec, stop - shift + 1, stop))
  reverse!(smallview(vec, start, stop))
  return vec
end

@inline function Base.insert!(vec::AbstractSmallVector, index::Integer, item)
  resize!(vec, length(vec) + 1)
  circshift!(smallview(vec, index, lastindex(vec)), 1)
  @inbounds vec[index] = item
  return vec
end

# Don't @inline, makes it slower.
function StaticArrays.insert(vec::AbstractSmallVector, index::Integer, item)
  mvec = thaw(vec)
  insert!(mvec, index, item)
  return convert(similar_type(vec), mvec)
end

@inline function Base.deleteat!(vec::AbstractSmallVector, index::Integer)
  circshift!(smallview(vec, index, lastindex(vec)), -1)
  resize!(vec, length(vec) - 1)
  return vec
end

@inline function Base.deleteat!(
  vec::AbstractSmallVector, indices::AbstractUnitRange{<:Integer}
)
  f = first(indices)
  n = length(indices)
  circshift!(smallview(vec, f, lastindex(vec)), -n)
  resize!(vec, length(vec) - n)
  return vec
end

# Don't @inline, makes it slower.
function StaticArrays.deleteat(
  vec::AbstractSmallVector, index::Union{Integer,AbstractUnitRange{<:Integer}}
)
  mvec = thaw(vec)
  deleteat!(mvec, index)
  return convert(similar_type(vec), mvec)
end

# InsertionSortAlg
# https://github.com/JuliaLang/julia/blob/bed2cd540a11544ed4be381d471bbf590f0b745e/base/sort.jl#L722-L736
# https://en.wikipedia.org/wiki/Insertion_sort#:~:text=Insertion%20sort%20is%20a%20simple,%2C%20heapsort%2C%20or%20merge%20sort.
# Alternatively could use `TupleTools.jl` or `StaticArrays.jl` for out-of-place sorting.
@inline function sort!(vec::AbstractSmallVector, order::Base.Sort.Ordering)
  lo, hi = firstindex(vec), lastindex(vec)
  lo_plus_1 = (lo + 1)
  @inbounds for i in lo_plus_1:hi
    j = i
    x = vec[i]
    jmax = j
    for _ in jmax:-1:lo_plus_1
      y = vec[j - 1]
      if !Base.Sort.lt(order, x, y)
        break
      end
      vec[j] = y
      j -= 1
    end
    vec[j] = x
  end
  return vec
end

@inline function Base.sort!(
  vec::AbstractSmallVector; lt=isless, by=identity, rev::Bool=false
)
  SmallVectors.sort!(vec, Base.Sort.ord(lt, by, rev))
  return vec
end

# Don't @inline, makes it slower.
function sort(vec::AbstractSmallVector, order::Base.Sort.Ordering)
  mvec = thaw(vec)
  SmallVectors.sort!(mvec, order)
  return convert(similar_type(vec), mvec)
end

@inline function Base.sort(
  vec::AbstractSmallVector; lt=isless, by=identity, rev::Bool=false
)
  return SmallVectors.sort(vec, Base.Sort.ord(lt, by, rev))
end

@inline function insertsorted!(vec::AbstractSmallVector, item; kwargs...)
  insert!(vec, searchsortedfirst(vec, item; kwargs...), item)
  return vec
end

function insertsorted(vec::AbstractSmallVector, item; kwargs...)
  mvec = thaw(vec)
  insertsorted!(mvec, item; kwargs...)
  return convert(similar_type(vec), mvec)
end

@inline function insertsortedunique!(vec::AbstractSmallVector, item; kwargs...)
  r = searchsorted(vec, item; kwargs...)
  if length(r) == 0
    insert!(vec, first(r), item)
  end
  return vec
end

# Code repeated since inlining doesn't work.
function insertsortedunique(vec::AbstractSmallVector, item; kwargs...)
  r = searchsorted(vec, item; kwargs...)
  if length(r) == 0
    vec = insert(vec, first(r), item)
  end
  return vec
end

@inline function mergesorted!(vec::AbstractSmallVector, item::AbstractVector; kwargs...)
  for x in item
    insertsorted!(vec, x; kwargs...)
  end
  return vec
end

function mergesorted(vec::AbstractSmallVector, item; kwargs...)
  mvec = thaw(vec)
  mergesorted!(mvec, item; kwargs...)
  return convert(similar_type(vec), mvec)
end

@inline function mergesortedunique!(
  vec::AbstractSmallVector, item::AbstractVector; kwargs...
)
  for x in item
    insertsortedunique!(vec, x; kwargs...)
  end
  return vec
end

# Code repeated since inlining doesn't work.
function mergesortedunique(vec::AbstractSmallVector, item; kwargs...)
  for x in item
    vec = insertsortedunique(vec, x; kwargs...)
  end
  return vec
end

Base.@propagate_inbounds function Base.copyto!(
  vec::AbstractSmallVector, item::AbstractVector
)
  for i in eachindex(item)
    vec[i] = item[i]
  end
  return vec
end

# Don't @inline, makes it slower.
function Base.circshift(vec::AbstractSmallVector, shift::Integer)
  mvec = thaw(vec)
  circshift!(mvec, shift)
  return convert(similar_type(vec), mvec)
end

@inline function Base.append!(vec::AbstractSmallVector, item::AbstractVector)
  l = length(vec)
  r = length(item)
  resize!(vec, l + r)
  @inbounds copyto!(smallview(vec, l + 1, l + r + 1), item)
  return vec
end

# Missing from `StaticArrays.jl`.
# Don't @inline, makes it slower.
function append(vec::AbstractSmallVector, item::AbstractVector)
  mvec = thaw(vec)
  append!(mvec, item)
  return convert(similar_type(vec), mvec)
end

@inline function Base.prepend!(vec::AbstractSmallVector, item::AbstractVector)
  l = length(vec)
  r = length(item)
  resize!(vec, l + r)
  circshift!(vec, length(item))
  @inbounds copyto!(vec, item)
  return vec
end

# Missing from `StaticArrays.jl`.
# Don't @inline, makes it slower.
function prepend(vec::AbstractSmallVector, item::AbstractVector)
  mvec = thaw(vec)
  prepend!(mvec, item)
  return convert(similar_type(vec), mvec)
end

# Don't @inline, makes it slower.
function smallvector_vcat(vec1::AbstractSmallVector, vec2::AbstractVector)
  mvec1 = thaw(vec1)
  append!(mvec1, vec2)
  return convert(similar_type(vec1), mvec1)
end

function Base.vcat(vec1::AbstractSmallVector{<:Number}, vec2::AbstractVector{<:Number})
  return smallvector_vcat(vec1, vec2)
end

Base.vcat(vec1::AbstractSmallVector, vec2::AbstractVector) = smallvector_vcat(vec1, vec2)
