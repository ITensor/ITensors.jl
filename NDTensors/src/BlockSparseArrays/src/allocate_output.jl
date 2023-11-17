#############################################################################
# Generic
#############################################################################

function output_type(f, args::Type...)
  # TODO: Is this good to use here?
  # Seems best for `Number` subtypes, maybe restrict to that here.
  return typeof(f(zero.(args)...))
  # return Base.promote_op(f, args...)
end

function output_type(f::Function, as::Type{<:AbstractArray}...)
  @assert allequal(ndims.(as))
  elt = output_type(f, eltype.(as)...)
  n = ndims(first(as))
  # TODO: Generalize this to GPU arrays!
  return Array{elt,n}
end

#############################################################################
# AbstractArray
#############################################################################

# Related to:
# https://github.com/JuliaLang/julia/issues/18161
# https://github.com/JuliaLang/julia/issues/25107
# https://github.com/JuliaLang/julia/issues/11557
abstract type AbstractArrayStructure{ElType,Axes} end

# TODO: Make this backwards compatible.
# TODO: Add a default for `eltype`.
# TODO: Change `Base.@kwdef` to `@kwdef`.
Base.@kwdef struct ArrayStructure{ElType,Axes} <: AbstractArrayStructure{ElType,Axes}
  eltype::ElType
  axes::Axes
end

function output_eltype(::typeof(map_nonzeros), fmap, as::Type{<:AbstractArray}...)
  return output_type(fmap, eltype.(as)...)
end

function output_eltype(f::typeof(map_nonzeros), fmap, as::AbstractArray...)
  # TODO: Compute based on runtime information?
  return output_eltype(f, fmap, typeof.(as)...)
end

function output_axes(f::typeof(map_nonzeros), fmap, as::AbstractArray...)
  # TODO: Make this more sophisticated, BlockSparseArrays
  # may have different block shapes.
  @assert allequal(axes.(as))
  return axes(first(as))
end

# Defaults to `ArrayStructure`.
# Maybe define a `default_output_structure`?
function output_structure(f::typeof(map_nonzeros), fmap, as::AbstractArray...)
  return ArrayStructure(;
    eltype=output_eltype(f, fmap, as...), axes=output_axes(f, fmap, as...)
  )
end

# Defaults to `ArrayStructure`.
# Maybe define a `default_output_type`?
function output_type(f::typeof(map_nonzeros), fmap, as::AbstractArray...)
  return Array
end

# Allocate an array with uninitialized/undefined memory
# according the array type and structure (for example the
# size or axes).
function allocate(arraytype::Type{<:AbstractArray}, structure)
  # TODO: Use `set_eltype`.
  return arraytype{structure.eltype}(undef, structure.axes)
end

function allocate_zeros(arraytype::Type{<:AbstractArray}, structure)
  a = allocate(arraytype, structure)
  # Assumes `arraytype` is mutable.
  # TODO: Use `zeros!!` or `zerovector!!` from VectorInterface.jl?
  map!(Returns(false), a, a)
  return a
end

function allocate_output(f::typeof(map_nonzeros), fmap, as::AbstractArray...)
  return allocate_zeros(output_type(f, fmap, as...), output_structure(f, fmap, as...))
end

#############################################################################
# SparseArray
#############################################################################

# TODO: Maybe store nonzero locations?
# TODO: Make this backwards compatible.
# TODO: Add a default for `eltype` and `zero`.
# TODO: Change `Base.@kwdef` to `@kwdef`.
Base.@kwdef struct SparseArrayStructure{ElType,Axes,Zero} <: AbstractArrayStructure{ElType,Axes}
  eltype::ElType
  axes::Axes
  zero::Zero
end

function allocate(arraytype::Type{<:SparseArray}, structure::SparseArrayStructure)
  # TODO: Use `set_eltype`.
  return arraytype{structure.eltype}(structure.axes, structure.zero)
end

function output_structure(f::typeof(map_nonzeros), fmap, as::SparseArrayLike...)
  return SparseArrayStructure(;
    eltype=output_eltype(f, fmap, as...),
    axes=output_axes(f, fmap, as...),
    zero=output_zero(f, fmap, as...),
  )
end

function output_type(f::typeof(map_nonzeros), fmap, as::SparseArrayLike...)
  return SparseArray
end

function output_zero(f::typeof(map_nonzeros), fmap, as::SparseArrayLike...)
  # TODO: Check they are all the same, update for now `axes`?
  return first(as).zero
end
