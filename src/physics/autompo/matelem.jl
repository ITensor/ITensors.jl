##################################
# MatElem (simple sparse matrix) #
##################################

struct MatElem{T}
  row::Int
  col::Int
  val::T
end

#function Base.show(io::IO,m::MatElem)
#  print(io,"($(m.row),$(m.col),$(m.val))")
#end

function toMatrix(els::Vector{MatElem{T}})::Matrix{T} where {T}
  nr = 0
  nc = 0
  for el in els
    nr = max(nr, el.row)
    nc = max(nc, el.col)
  end
  M = zeros(T, nr, nc)
  for el in els
    M[el.row, el.col] = el.val
  end
  return M
end

function Base.:(==)(m1::MatElem{T}, m2::MatElem{T})::Bool where {T}
  return (m1.row == m2.row && m1.col == m2.col && m1.val == m2.val)
end

function Base.isless(m1::MatElem{T}, m2::MatElem{T})::Bool where {T}
  if m1.row != m2.row
    return m1.row < m2.row
  elseif m1.col != m2.col
    return m1.col < m2.col
  end
  return m1.val < m2.val
end
