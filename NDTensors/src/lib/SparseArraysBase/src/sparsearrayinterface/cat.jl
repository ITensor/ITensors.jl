unval(x) = x
unval(::Val{x}) where {x} = x

# TODO: Assert that `a1` and `a2` start at one.
axis_cat(a1::AbstractUnitRange, a2::AbstractUnitRange) = Base.OneTo(length(a1) + length(a2))
function axis_cat(
  a1::AbstractUnitRange, a2::AbstractUnitRange, a_rest::AbstractUnitRange...
)
  return axis_cat(axis_cat(a1, a2), a_rest...)
end
function cat_axes(as::AbstractArray...; dims)
  return ntuple(length(first(axes.(as)))) do dim
    return if dim in unval(dims)
      axis_cat(map(axes -> axes[dim], axes.(as))...)
    else
      axes(first(as))[dim]
    end
  end
end

function allocate_cat_output(as::AbstractArray...; dims)
  eltype_dest = promote_type(eltype.(as)...)
  axes_dest = cat_axes(as...; dims)
  # TODO: Promote the block types of the inputs rather than using
  # just the first input.
  # TODO: Make this customizable with `cat_similar`.
  # TODO: Base the zero element constructor on those of the inputs,
  # for example block sparse arrays.
  return similar(first(as), eltype_dest, axes_dest...)
end

# https://github.com/JuliaLang/julia/blob/v1.11.1/base/abstractarray.jl#L1748-L1857
# https://docs.julialang.org/en/v1/base/arrays/#Concatenation-and-permutation
# This is very similar to the `Base.cat` implementation but handles zero values better.
function cat_offset!(
  a_dest::AbstractArray, offsets, a1::AbstractArray, a_rest::AbstractArray...; dims
)
  inds = ntuple(ndims(a_dest)) do dim
    dim in unval(dims) ? offsets[dim] .+ axes(a1, dim) : axes(a_dest, dim)
  end
  a_dest[inds...] = a1
  new_offsets = ntuple(ndims(a_dest)) do dim
    dim in unval(dims) ? offsets[dim] + size(a1, dim) : offsets[dim]
  end
  cat_offset!(a_dest, new_offsets, a_rest...; dims)
  return a_dest
end
function cat_offset!(a_dest::AbstractArray, offsets; dims)
  return a_dest
end

# TODO: Define a generic `cat!` function.
function sparse_cat!(a_dest::AbstractArray, as::AbstractArray...; dims)
  offsets = ntuple(zero, ndims(a_dest))
  # TODO: Fill `a_dest` with zeros if needed.
  cat_offset!(a_dest, offsets, as...; dims)
  return a_dest
end

function sparse_cat(as::AbstractArray...; dims)
  a_dest = allocate_cat_output(as...; dims)
  sparse_cat!(a_dest, as...; dims)
  return a_dest
end
