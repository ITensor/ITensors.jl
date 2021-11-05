module LazyApply

using Zeros

import Base: *, +, -, /, exp, adjoint, show, ==, convert, getindex, length, iterate

export coefficient

struct Applied{F,Args}
  f::F
  args::Args
  function Applied{F,Args}(f, args::Tuple) where {F,Args}
    return new{F,Args}(f, args)
  end
end
Applied(f, args::Tuple) = Applied{typeof(f), typeof(args)}(f, args)
Applied(f, args...) = Applied(f, args)

# TODO: This makes shorthands like `Add(1, 2)` work, but probably
# it is bad to use `F.instance` to get the function from the type.
Applied{F,Args}(args::Tuple) where {F,Args} = Applied{F,Args}(F.instance, args)
Applied{F,Args}(args...) where {F,Args} = Applied{F,Args}(args)
Applied{F}(args::Tuple) where {F} = Applied{F,typeof(args)}(args)
Applied{F}(args...) where {F} = Applied{F}(args)

_empty(::Type{T}) where {T} = error("_empty not implemented for type $T.")
_empty(::Type{Tuple{T}}) where {T} = (_empty(T),)
_empty(::Type{Vector{T}}) where {T} = Vector{T}()

function Applied{F,Args}() where {F,Args}
  return Applied{F,Args}(_empty(Args))
end

function (arg1::Applied == arg2::Applied)
  return (arg1.f == arg2.f) && (arg1.args == arg2.args)
end

# Shorthands
const Scaled{A,B} = Applied{typeof(*),Tuple{A,B}} where {A<:Number}
const Add{T} = Applied{typeof(+),T}
const Sum{T} = Applied{typeof(sum),T}
const Prod{T} = Applied{typeof(prod),T}
# TODO: Why does the order of the type parameters need to be switched?
const ScaledProd{A,T} = Scaled{Prod{T},A}
const Mul{T} = Applied{typeof(*),T}
const ∑ = Sum
const ∏ = Prod

const Exp{T} = Applied{typeof(exp),T}

coefficient(arg::Applied) = 𝟏

const SumOrProd{T} = Union{Sum{T},Prod{T}}

length(arg::SumOrProd) = length(arg.args...)
getindex(arg::SumOrProd, n) = getindex(arg.args..., n)
iterate(arg::SumOrProd, args...) = iterate(arg.args..., args...)

length(arg::Scaled) = length(arg.args[2])
getindex(arg::Scaled, n) = getindex(arg.args[2], n)
coefficient(arg::Scaled) = arg.args[1]
iterate(arg::Scaled, args...) = iterate(arg.args[2], args...)

Base.convert(::Type{Applied{F,Args}}, arg::Applied{F,Args}) where {F,Args} = arg

# For some reasons this conversion isn't being done automatically.
function convert(::Type{Applied{F,Args1}}, arg2::Applied{F,Args2}) where {F,Args1,Args2}
  return Applied{F,Args1}(arg2.f, convert(Args1, arg2.args))
end

function try_promote(x::T, y::S) where {T,S}
  R = promote_type(T, S)
  return (convert(R, x), convert(R, y))
end

# Conversion
Sum(arg::Add) = Sum(collect(arg.args))

# Scalar multiplication (general rules)
_mul(arg1::Number, arg2) = Mul(arg1, arg2)
(arg1::Number * arg2::Applied) = _mul(arg1, arg2)
(arg1::Number * arg2::Prod) = _mul(arg1, arg2)

# Scalar division
(arg1::Applied / arg2::Number) = inv(arg2) * arg1

# Put the scalar value first by convention
_mul(arg1, arg2::Number) = Mul(arg2, arg1)
(arg1::Applied * arg2::Number) = _mul(arg1, arg2)
(arg1::Prod * arg2::Number) = _mul(arg1, arg2)

# Scalar multiplication (specialized rules)
(arg1::Number * arg2::Scaled) = Mul(arg1 * arg2.args[1], arg2.args[2])
(arg1::Scaled * arg2::Scaled) = Mul(arg1.args[1] * arg2.args[1], arg1.args[2] * arg2.args[2])
(arg1::Scaled * arg2) = Mul(arg1.args[1], arg1.args[2] * arg2)
(arg1 * arg2::Scaled) = Mul(arg2.args[1], arg1 * arg2.args[2])
# Scalars are treated special for the sake of multiplication
Mul(arg1::Number, arg2::Number) = arg1 * arg2
Mul(arg1::Number, arg2::Scaled) = arg1 * arg2
Mul(arg1::Scaled, arg2::Number) = arg1 * arg2
(arg1::Number * arg2::Sum) = Sum(Mul.(arg1, arg2))
(arg1::Number * arg2::Add) = Add(Mul.(arg1, arg2.args))

# Types should implement `__sum`.
_sum(arg1, arg2) = __sum(try_promote(arg1, arg2)...)

# Addition (general rules)
__sum(arg1, arg2) = Sum(vcat(arg1, arg2))
(arg1::Applied + arg2::Applied) = _sum(arg1, arg2)
(arg1::Applied + arg2) = _sum(arg1, arg2)
(arg1 + arg2::Applied) = _sum(arg1, arg2)

# Subtraction (general rules)
_subtract(arg1, arg2) = _sum(arg1, Mul(-𝟏, arg2))
(arg1::Applied - arg2::Applied) = _subtract(arg1, arg2)
(arg1::Applied - arg2) = _subtract(arg1, arg2)
(arg1 - arg2::Applied) = _subtract(arg1, arg2)

# Addition (specialized rules)
__sum(arg1::Sum, arg2::Sum) = Sum(vcat(arg1.args..., arg2.args...))
(arg1::Sum + arg2::Sum) = _sum(arg1, arg2)
(arg1::Add + arg2::Add) = Add(arg1.args..., arg2.args...)

__sum(arg1::Sum, arg2) = Sum(vcat(arg1.args..., arg2))
__sum(arg1, arg2::Sum) = Sum(vcat(arg1, arg2.args...))
(arg1::Sum + arg2) = _sum(arg1, arg2)
(arg1::Sum + arg2::Applied) = _sum(arg1, arg2)

(arg1 + arg2::Sum) = _sum(arg1, arg2)
(arg1::Add + arg2) = Add(arg1.args..., arg2)
(arg1 + arg2::Add) = Add(arg1, arg2.args...)

# Multiplication (general rules)
(arg1::Applied * arg2::Applied) = Mul(arg1, arg2)

# Multiplication (specialized rules)
(arg1::Prod * arg2::Prod) = Prod(vcat(arg1.args..., arg2.args...))
(arg1::Sum * arg2::Sum) = Prod([arg1, arg2])

_prod(arg1::Prod, arg2) = Prod(vcat(arg1.args..., arg2))
(arg1::Prod * arg2) = _prod(arg1, arg2)
(arg1::Prod * arg2::Applied) = _prod(arg1, arg2)

_prod(arg1, arg2::Prod) = Prod(vcat(arg1, arg2...))
(arg1 * arg2::Prod) = _prod(arg1, arg2)
(arg1::Applied * arg2::Prod) = _prod(arg1, arg2)

# Other lazy operations
exp(arg::Applied) = Applied(exp, arg)

# adjoint
adjoint(arg::Applied) = Applied(adjoint, arg)
adjoint(arg::Applied{typeof(adjoint)}) = only(arg.args)
adjoint(arg::Prod) = ∏(reverse(adjoint.(arg)))

# Materialize
materialize(a::Number) = a
materialize(a::AbstractString) = a
materialize(a::Vector) = materialize.(a)
materialize(a::Applied) = a.f(materialize.(a.args)...)

_print(io::IO, args...) = print(io, args...)
function _print(io::IO, a::AbstractVector, args...)
  print(io, "[")
  for n in 1:length(a)
    _print(io, a[n], args...)
    if n < length(a)
      print(io, ",\n")
    end
  end
  print(io, "]")
end

function show(io::IO, m::MIME"text/plain", a::Applied)
  print(io, a.f, "(\n")
  for n in 1:length(a.args)
    _print(io, a.args[n])
    if n < length(a.args)
      print(io, ", ")
    end
  end
  print(io, "\n)")
end
show(io::IO, a::Applied) = show(io, MIME("text/plain"), a)
end
