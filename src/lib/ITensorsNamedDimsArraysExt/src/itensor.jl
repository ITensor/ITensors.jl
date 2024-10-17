using ITensors: ITensors
using Random: AbstractRNG, default_rng

# Constructors
ITensors.ITensor(na::AbstractNamedDimsArray) = ITensors._ITensor(na)
ITensors.itensor(na::AbstractNamedDimsArray) = ITensors._ITensor(na)

# Convenient constructors
default_eltype() = Float64
for f in [:rand, :randn]
  @eval begin
    function Base.$f(
      rng::AbstractRNG, elt::Type{<:Number}, dims::Tuple{Index,Vararg{Index}}
    )
      return ITensor($f(rng, elt, NamedInt.(dims)))
    end
    function Base.$f(
      rng::AbstractRNG, elt::Type{<:Number}, dim1::Index, dims::Vararg{Index}
    )
      return $f(rng, elt, (dim1, dims...))
    end
    Base.$f(elt::Type{<:Number}, dims::Tuple{Index,Vararg{Index}}) = $f(
      default_rng(), elt, dims
    )
    Base.$f(elt::Type{<:Number}, dim1::Index, dims::Vararg{Index}) = $f(
      elt, (dim1, dims...)
    )
    Base.$f(dims::Tuple{Index,Vararg{Index}}) = $f(default_eltype(), dims)
    Base.$f(dim1::Index, dims::Vararg{Index}) = $f((dim1, dims...))
  end
end
for f in [:zeros, :ones]
  @eval begin
    function Base.$f(elt::Type{<:Number}, dims::Tuple{Index,Vararg{Index}})
      return ITensor($f(elt, NamedInt.(dims)))
    end
    function Base.$f(elt::Type{<:Number}, dim1::Index, dims::Vararg{Index})
      return $f(elt, (dim1, dims...))
    end
    Base.$f(dims::Tuple{Index,Vararg{Index}}) = $f(default_eltype(), dims)
    Base.$f(dim1::Index, dims::Vararg{Index}) = $f((dim1, dims...))
  end
end
function Base.fill(value, dims::Tuple{Index,Vararg{Index}})
  return ITensor(fill(value, NamedInt.(dims)))
end
function Base.fill(value, dim1::Index, dims::Vararg{Index})
  return fill(value, (dim1, dims...))
end
