- Logic around element type promotion, BangBang.jl syntax.
```julia
module MutableStorageArrays

module NoBang
function setindex(a::AbstractArray, value, I::Int...)
  a′ = similar(a, typeof(value))
  a′ .= a
  a′[I...] = value
  return a′
end

function setindex(a::AbstractArray, value, I...)
  a′ = similar(a, eltype(value))
  a′ .= a
  a′[I...] = value
  return a′
end
end

function may(f!, args...)
  if possible(f!, args...)
    return f!(args...)
  end
  return pure(f!)(args...)
end

implements(f!, x) = implements(f!, typeof(x))
implements(f!, ::Type) = false

function setindex!! end
setindex!!(a::AbstractArray, value, I...) = may(setindex!, a, value, I...)
implements(::typeof(setindex!), ::Type{<:AbstractArray}) = true
pure(::typeof(setindex!)) = NoBang.setindex
maymutate(::typeof(setindex!)) = setindex!!
function possible(::typeof(setindex!), a::AbstractArray, value, I::Int...)
  return implements(setindex!, a) && promote_type(eltype(a), typeof(value)) <: eltype(a)
end
function possible(::typeof(setindex!), a::AbstractArray, value, I...)
  return implements(setindex!, a) && promote_type(eltype(a), eltype(value)) <: eltype(a)
end

struct MutableStorageArrayInterface <: AbstractArrayInterface end

# Minimal interface.
function storage end
function setstorage! end

@interface ::MutableStorageArrayInterface function Base.setindex!(a::AbstractArray, value, I...)
  return setstorage!(a, storage(setindex!!(a, value, I...)))
end

end
```
