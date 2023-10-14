function mul!!(::Type{<:Array}, CM, ::Type{<:Array}, AM, ::Type{<:Array}, BM, α, β)
  @strided CM = mul!(CM, AM, BM, α, β)
  return CM
end
