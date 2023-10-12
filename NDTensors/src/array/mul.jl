function mul!!(::Type{<:Array}, CM, ::Type{<:Array}, AM, ::Type{<:Array}, BM, α, β)
  return @strided mul!(CM, AM, BM, α, β)
end
