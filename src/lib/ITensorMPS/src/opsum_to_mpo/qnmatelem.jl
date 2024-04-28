struct QNMatElem{T}
  rowqn::QN
  colqn::QN
  row::Int
  col::Int
  val::T
end

function Base.:(==)(m1::QNMatElem{T}, m2::QNMatElem{T})::Bool where {T}
  return (
    m1.row == m2.row &&
    m1.col == m2.col &&
    m1.val == m2.val &&
    m1.rowqn == m2.rowqn &&
    m1.colqn == m2.colqn
  )
end

function Base.isless(m1::QNMatElem{T}, m2::QNMatElem{T})::Bool where {T}
  if m1.rowqn != m2.rowqn
    return m1.rowqn < m2.rowqn
  elseif m1.colqn != m2.colqn
    return m1.colqn < m2.colqn
  elseif m1.row != m2.row
    return m1.row < m2.row
  elseif m1.col != m2.col
    return m1.col < m2.col
  end
  return m1.val < m2.val
end
