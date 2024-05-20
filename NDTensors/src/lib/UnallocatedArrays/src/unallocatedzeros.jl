using FillArrays:
  FillArrays,
  AbstractZeros,
  Fill,
  Zeros,
  broadcasted_fill,
  broadcasted_zeros,
  kron_fill,
  kron_zeros,
  mult_zeros

## TODO Should Alloc also be of ElT and N or should there be
## More freedom there?
struct UnallocatedZeros{ElT,N,Axes,Alloc} <: AbstractZeros{ElT,N,Axes}
  z::Zeros{ElT,N,Axes}
  alloc::Alloc
end

function UnallocatedZeros{ElT,N,Axes}(z::Zeros, alloc::Type) where {ElT,N,Axes}
  return UnallocatedZeros{ElT,N,Axes,Type{alloc}}(z, alloc)
end

function UnallocatedZeros{ElT,N}(z::Zeros, alloc) where {ElT,N}
  return UnallocatedZeros{ElT,N,typeof(axes(z))}(z, alloc)
end

function UnallocatedZeros{ElT}(z::Zeros, alloc) where {ElT}
  return UnallocatedZeros{ElT,ndims(z)}(z, alloc)
end

set_alloctype(f::Zeros, alloc::Type) = UnallocatedZeros(f, alloc)

Base.parent(Z::UnallocatedZeros) = Z.z

Base.convert(::Type{<:UnallocatedZeros}, A::UnallocatedZeros) = A

#############################################
# Arithmatic

function FillArrays.mult_zeros(a::UnallocatedZeros, b, elt, ax)
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end
FillArrays.mult_zeros(a, b::UnallocatedZeros, elt, ax) = mult_zeros(b, a, elt, ax)
function FillArrays.mult_zeros(a::UnallocatedZeros, b::UnallocatedZeros, elt, ax)
  @assert alloctype(a) == alloctype(b)
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end

function FillArrays.broadcasted_zeros(f, a::UnallocatedZeros, elt, ax)
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end
function FillArrays.broadcasted_zeros(f, a::UnallocatedZeros, b::UnallocatedZeros, elt, ax)
  @assert alloctype(a) == alloctype(b)
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end

function FillArrays.broadcasted_zeros(f, a::UnallocatedZeros, b, elt, ax)
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end
function FillArrays.broadcasted_zeros(f, a, b::UnallocatedZeros, elt, ax)
  return broadcasted_zeros(f, b, a, elt, ax)
end

function FillArrays.broadcasted_fill(f, a::UnallocatedZeros, val, ax)
  return UnallocatedFill(Fill(val, ax), alloctype(a))
end
function FillArrays.broadcasted_fill(f, a::UnallocatedZeros, b, val, ax)
  return UnallocatedFill(Fill(val, ax), alloctype(a))
end

function FillArrays.broadcasted_fill(f, a, b::UnallocatedZeros, val, ax)
  return broadcasted_fill(f, b, a, val, ax)
end

function FillArrays.kron_zeros(a::UnallocatedZeros, b::UnallocatedZeros, elt, ax)
  @assert alloctype(a) == alloctype(b)
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end

function FillArrays.kron_fill(a::UnallocatedZeros, b::UnallocatedFill, val, ax)
  @assert alloctype(a) == alloctype(b)
  elt = typeof(val)
  return UnallocatedZeros(Zeros{elt}(ax), alloctype(a))
end

function FillArrays.kron_fill(a::UnallocatedFill, b::UnallocatedZeros, val, ax)
  return kron_fill(b, a, val, ax)
end
