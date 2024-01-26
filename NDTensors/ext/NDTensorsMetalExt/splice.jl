function Base.splice!(e::Exposed{<:MtlArray}, indices, replacement...)
  error("Not implemented")
  a = unexpose(e)
  res = splice!(a, indices, replacement...)
  return res
end
