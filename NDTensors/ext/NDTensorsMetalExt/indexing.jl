function Base.getindex(E::Exposed{<:MtlArray})
  return Metal.@allowscalar getindex(unexpose(E))
end

function Base.setindex!(E::Exposed{<:MtlArray}, x::Number)
  Metal.@allowscalar setindex!(unexpose(E), x)
  return unexpose(E)
end
