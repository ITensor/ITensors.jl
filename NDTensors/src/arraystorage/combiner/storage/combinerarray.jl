struct CombinerArray{N,C,Axes<:Tuple{Vararg{Any,N}}} <: AbstractArray{Any,N}
  combiner::C
  axes::Axes
end

Base.axes(a::CombinerArray) = a.axes
Base.size(a::CombinerArray) = length.(axes(a))
Base.getindex(a::CombinerArray{0}) = true

function Base.conj(aliasstyle::AliasStyle, a::CombinerArray)
  return CombinerArray(conj(aliasstyle, a.combiner), axes(a))
end
