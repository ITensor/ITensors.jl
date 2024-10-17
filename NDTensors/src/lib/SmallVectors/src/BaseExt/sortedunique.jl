# Union two unique sorted collections into an
# output buffer, returning a unique sorted collection.

using Base: Ordering, ord, lt

function unionsortedunique!(
  itr1,
  itr2;
  lt=isless,
  by=identity,
  rev::Union{Bool,Nothing}=nothing,
  order::Ordering=Forward,
)
  return unionsortedunique!(itr1, itr2, ord(lt, by, rev, order))
end

function unionsortedunique!(itr1, itr2, order::Ordering)
  i1 = firstindex(itr1)
  i2 = firstindex(itr2)
  stop1 = lastindex(itr1)
  stop2 = lastindex(itr2)
  @inbounds while i1 ≤ stop1 && i2 ≤ stop2
    item1 = itr1[i1]
    item2 = itr2[i2]
    if lt(order, item1, item2)
      i1 += 1
    elseif lt(order, item2, item1)
      # TODO: Use `insertat!`?
      resize!(itr1, length(itr1) + 1)
      for j in length(itr1):-1:(i1+1)
        itr1[j] = itr1[j - 1]
      end
      # Replace with the item from the second list
      itr1[i1] = item2
      i1 += 1
      i2 += 1
      stop1 += 1
    else # They are equal
      i1 += 1
      i2 += 1
    end
  end
  # TODO: Use `insertat!`?
  resize!(itr1, length(itr1) + (stop2 - i2 + 1))
  @inbounds for j2 in i2:stop2
    itr1[i1] = itr2[j2]
    i1 += 1
  end
  return itr1
end

function unionsortedunique(
  itr1,
  itr2;
  lt=isless,
  by=identity,
  rev::Union{Bool,Nothing}=nothing,
  order::Ordering=Forward,
)
  return unionsortedunique(itr1, itr2, ord(lt, by, rev, order))
end

# Union two unique sorted collections into an
# output buffer, returning a unique sorted collection.
function unionsortedunique(itr1, itr2, order::Ordering)
  out = thaw_type(itr1)()
  i1 = firstindex(itr1)
  i2 = firstindex(itr2)
  iout = firstindex(out)
  stop1 = lastindex(itr1)
  stop2 = lastindex(itr2)
  stopout = lastindex(out)
  @inbounds while i1 ≤ stop1 && i2 ≤ stop2
    iout > stopout && resize!(out, iout)
    item1 = itr1[i1]
    item2 = itr2[i2]
    if lt(order, item1, item2)
      out[iout] = item1
      iout += 1
      i1 += 1
    elseif lt(order, item2, item1)
      out[iout] = item2
      iout += 1
      i2 += 1
    else # They are equal
      out[iout] = item2
      iout += 1
      i1 += 1
      i2 += 1
    end
  end
  # TODO: Use `insertat!`?
  r1 = i1:stop1
  resize!(out, length(out) + length(r1))
  @inbounds for j1 in r1
    out[iout] = itr1[j1]
    iout += 1
  end
  # TODO: Use `insertat!`?
  r2 = i2:stop2
  resize!(out, length(out) + length(r2))
  @inbounds for j2 in r2
    out[iout] = itr2[j2]
    iout += 1
  end
  return freeze(out)
end

function setdiffsortedunique!(
  itr1,
  itr2;
  lt=isless,
  by=identity,
  rev::Union{Bool,Nothing}=nothing,
  order::Ordering=Forward,
)
  return setdiffsortedunique!(itr1, itr2, ord(lt, by, rev, order))
end

function setdiffsortedunique!(itr1, itr2, order::Ordering)
  i1 = firstindex(itr1)
  i2 = firstindex(itr2)
  stop1 = lastindex(itr1)
  stop2 = lastindex(itr2)
  @inbounds while i1 ≤ stop1 && i2 ≤ stop2
    item1 = itr1[i1]
    item2 = itr2[i2]
    if lt(order, item1, item2)
      i1 += 1
    elseif lt(order, item2, item1)
      i2 += 1
    else # They are equal
      # TODO: Use `deletate!`?
      for j1 in i1:(length(itr1)-1)
        itr1[j1] = itr1[j1 + 1]
      end
      resize!(itr1, length(itr1) - 1)
      stop1 = lastindex(itr1)
      i2 += 1
    end
  end
  return itr1
end

function setdiffsortedunique(
  itr1,
  itr2;
  lt=isless,
  by=identity,
  rev::Union{Bool,Nothing}=nothing,
  order::Ordering=Forward,
)
  return setdiffsortedunique(itr1, itr2, ord(lt, by, rev, order))
end

function setdiffsortedunique(itr1, itr2, order::Ordering)
  out = thaw_type(itr1)()
  i1 = firstindex(itr1)
  i2 = firstindex(itr2)
  iout = firstindex(out)
  stop1 = lastindex(itr1)
  stop2 = lastindex(itr2)
  stopout = lastindex(out)
  @inbounds while i1 ≤ stop1 && i2 ≤ stop2
    item1 = itr1[i1]
    item2 = itr2[i2]
    if lt(order, item1, item2)
      iout > stopout && resize!(out, iout)
      out[iout] = item1
      iout += 1
      i1 += 1
    elseif lt(order, item2, item1)
      i2 += 1
    else # They are equal
      i1 += 1
      i2 += 1
    end
  end
  resize!(out, iout - 1)
  return freeze(out)
end

function intersectsortedunique!(
  itr1,
  itr2;
  lt=isless,
  by=identity,
  rev::Union{Bool,Nothing}=nothing,
  order::Ordering=Forward,
)
  return intersectsortedunique!(itr1, itr2, ord(lt, by, rev, order))
end

function intersectsortedunique!(itr1, itr2, order::Ordering)
  return error("Not implemented")
end

function intersectsortedunique(
  itr1,
  itr2;
  lt=isless,
  by=identity,
  rev::Union{Bool,Nothing}=nothing,
  order::Ordering=Forward,
)
  return intersectsortedunique(itr1, itr2, ord(lt, by, rev, order))
end

function intersectsortedunique(itr1, itr2, order::Ordering)
  return error("Not implemented")
end

function symdiffsortedunique!(
  itr1,
  itr2;
  lt=isless,
  by=identity,
  rev::Union{Bool,Nothing}=nothing,
  order::Ordering=Forward,
)
  return symdiffsortedunique!(itr1, itr2, ord(lt, by, rev, order))
end

function symdiffsortedunique!(itr1, itr2, order::Ordering)
  return error("Not implemented")
end

function symdiffsortedunique(
  itr1,
  itr2;
  lt=isless,
  by=identity,
  rev::Union{Bool,Nothing}=nothing,
  order::Ordering=Forward,
)
  return symdiffsortedunique(itr1, itr2, ord(lt, by, rev, order))
end

function symdiffsortedunique(itr1, itr2, order::Ordering)
  return error("Not implemented")
end
