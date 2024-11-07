# LabelledUnitRangeDual is obtained by slicing a GradedUnitRangeDual with a block

using ..LabelledNumbers: LabelledNumbers, label, labelled, unlabel

struct LabelledUnitRangeDual{T,NondualUnitRange<:AbstractUnitRange{T}} <:
       AbstractUnitRange{T}
  nondual_unitrange::NondualUnitRange
end

dual(a::LabelledUnitRange) = LabelledUnitRangeDual(a)
nondual(a::LabelledUnitRangeDual) = a.nondual_unitrange
dual(a::LabelledUnitRangeDual) = nondual(a)
flip(a::LabelledUnitRangeDual) = dual(flip(nondual(a)))
isdual(::LabelledUnitRangeDual) = true

LabelledNumbers.label(a::LabelledUnitRangeDual) = dual(label(nondual(a)))
LabelledNumbers.unlabel(a::LabelledUnitRangeDual) = unlabel(nondual(a))

for f in [:first, :getindex, :last, :length, :step]
  @eval Base.$f(a::LabelledUnitRangeDual, args...) =
    labelled($f(unlabel(a), args...), label(a))
end

# fix ambiguities
Base.getindex(a::LabelledUnitRangeDual, i::Integer) = dual(nondual(a)[i])

function Base.show(
  io::IO, ::MIME"text/plain", a::Union{LabelledUnitRange,LabelledUnitRangeDual}
)
  println(io, typeof(a))
  return print(io, label(a), " => ", unlabel(a))
end

function Base.show(io::IO, a::Union{LabelledUnitRange,LabelledUnitRangeDual})
  return print(io, nameof(typeof(a)), " ", label(a), " => ", unlabel(a))
end

function Base.AbstractUnitRange{T}(a::LabelledUnitRangeDual) where {T}
  return AbstractUnitRange{T}(nondual(a))
end
