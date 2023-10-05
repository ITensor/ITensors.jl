# TODO:
# Add `
# Version that uses an `Ordering`.
function _insorted(
  x,
  v::AbstractVector;
  lt=isless,
  by=identity,
  rev::Union{Bool,Nothing}=nothing,
  order::Ordering=Forward,
)
  return _insorted(x, v, ord(lt, by, rev, order))
end
_insorted(x, v::AbstractVector, o::Ordering) = !isempty(searchsorted(v, x, o))

function alluniquesorted(
  vec; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward
)
  return alluniquesorted(vec, ord(lt, by, rev, order))
end

function alluniquesorted(vec, order::Ordering)
  length(vec) < 2 && return true
  iter = eachindex(vec)
  I = iterate(iter)
  while I !== nothing
    i, s = I
    J = iterate(iter, s)
    isnothing(J) && return true
    j, _ = J
    !lt(order, @inbounds(vec[i]), @inbounds(vec[j])) && return false
    I = J
  end
  return true
end

function uniquesorted(vec; lt=isless, by=identity, rev::Bool=false, order::Ordering=Forward)
  return uniquesorted(vec, ord(lt, by, rev, order))
end

function uniquesorted(vec::AbstractVector, order::Ordering)
  vec = copy(vec)
  i = firstindex(vec)
  stopi = lastindex(vec)
  while i < stopi
    if !lt(order, @inbounds(vec[i]), @inbounds(vec[i + 1]))
      deleteat!(vec, i)
      stopi -= 1
    else
      i += 1
    end
  end
  return vec
end
