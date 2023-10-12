function NDTensors.mul!!(
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