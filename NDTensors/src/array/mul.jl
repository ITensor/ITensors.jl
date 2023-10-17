function mul!!(::Type{<:Array}, CM, ::Type{<:Array}, AM, ::Type{<:Array}, BM, α, β)
  @strided mul!(CM, AM, BM, α, β)
  return CM
end
