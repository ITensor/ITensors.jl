using Random: AbstractRNG, default_rng

# TODO: Use `AbstractNamedUnitRange`, determine the `AbstractNamedDimsArray`
# from a default value. Useful for distinguishing between `NamedDimsArray`
# and `ITensor`.
# Convenient constructors
default_eltype() = Float64
for f in [:rand, :randn]
  @eval begin
    function Base.$f(
      rng::AbstractRNG, elt::Type{<:Number}, dims::Tuple{NamedInt,Vararg{NamedInt}}
    )
      a = $f(rng, elt, unname.(dims))
      return named(a, name.(dims))
    end
    function Base.$f(
      rng::AbstractRNG, elt::Type{<:Number}, dim1::NamedInt, dims::Vararg{NamedInt}
    )
      return $f(rng, elt, (dim1, dims...))
    end
    Base.$f(elt::Type{<:Number}, dims::Tuple{NamedInt,Vararg{NamedInt}}) = $f(
      default_rng(), elt, dims
    )
    Base.$f(elt::Type{<:Number}, dim1::NamedInt, dims::Vararg{NamedInt}) = $f(
      elt, (dim1, dims...)
    )
    Base.$f(dims::Tuple{NamedInt,Vararg{NamedInt}}) = $f(default_eltype(), dims)
    Base.$f(dim1::NamedInt, dims::Vararg{NamedInt}) = $f((dim1, dims...))
  end
end
for f in [:zeros, :ones]
  @eval begin
    function Base.$f(elt::Type{<:Number}, dims::Tuple{NamedInt,Vararg{NamedInt}})
      a = $f(elt, unname.(dims))
      return named(a, name.(dims))
    end
    function Base.$f(elt::Type{<:Number}, dim1::NamedInt, dims::Vararg{NamedInt})
      return $f(elt, (dim1, dims...))
    end
    Base.$f(dims::Tuple{NamedInt,Vararg{NamedInt}}) = $f(default_eltype(), dims)
    Base.$f(dim1::NamedInt, dims::Vararg{NamedInt}) = $f((dim1, dims...))
  end
end
function Base.fill(value, dims::Tuple{NamedInt,Vararg{NamedInt}})
  a = fill(value, unname.(dims))
  return named(a, name.(dims))
end
function Base.fill(value, dim1::NamedInt, dims::Vararg{NamedInt})
  return fill(value, (dim1, dims...))
end
