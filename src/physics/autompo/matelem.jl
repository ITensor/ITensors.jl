##################################
# MatElem (simple sparse matrix) #
##################################

struct MatElem{T}
  row::Int
  col::Int
  val::T
end

eltype(::Type{<:MatElem{T}}) where {T} = T
eltype(m::MatElem) = eltype(typeof(m))

function to_matrix(els::Vector{<:MatElem})
  nr, nc = 0, 0
  for el in els
    nr = max(nr, el.row)
    nc = max(nc, el.col)
  end
  M = zeros(eltype(eltype(els)), nr, nc)
  for el in els
    M[el.row, el.col] = el.val
  end
  return M
end

function (m1::MatElem == m2::MatElem)
  return (m1.row == m2.row && m1.col == m2.col && m1.val == m2.val)
end

function isless(m1::MatElem, m2::MatElem)
  if m1.row != m2.row
    return m1.row < m2.row
  elseif m1.col != m2.col
    return m1.col < m2.col
  end
  return m1.val < m2.val
end
