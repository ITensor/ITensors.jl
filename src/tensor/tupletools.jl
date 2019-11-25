export insertat,
       deleteat,
       tuplecat

# Combine a bunch of tuples
# TODO: move this functionality to IndexSet, combine with unioninds?
@inline tuplecat(x) = x
@inline tuplecat(x, y) = (x..., y...)
@inline tuplecat(x, y, z...) = (x..., tuplecat(y, z...)...)

#function tuplecat(is1::NTuple{N1},
#                  is2::NTuple{N2}) where {N1,N2}
#  return tuple(is1...,is2...)
#end

function _deleteat(i,pos)
  i < pos && return t[i]
  return t[i+1]
end

function deleteat(t::NTuple{N},pos::Int) where {N}
  return ntuple(i -> _deleteat(i,pos),Val(N-1))
end

function _insertat(i,pos,n_insert)
  if i < pos
    return t[i]
  elseif i > pos+n_insert-1
    return t[i-1]
  end
  return val[i-pos+1]
end

function insertat(t::NTuple{N},
                  val::NTuple{M},
                  pos::Int) where {N,M}
  return ntuple(i -> _insertat(i,pos,M),Val(N+M-1))
end

