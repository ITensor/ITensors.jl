function mul!!(CM::AbstractArray, AM::AbstractArray, BM::AbstractArray, α, β)
  return mul!!(leaf_parenttype(CM), CM, leaf_parenttype(AM), AM, leaf_parenttype(BM), BM, α, β)
end

function mul!!(
  ::Type{<:AbstractArray},
  CM,
  ::Type{<:AbstractArray},
  AM,
  ::Type{<:AbstractArray},
  BM,
  α,
  β,
)
  return mul!(CM, AM, BM, α, β)
end