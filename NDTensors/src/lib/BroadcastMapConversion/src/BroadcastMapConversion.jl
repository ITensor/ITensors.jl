module BroadcastMapConversion
# Convert broadcast call to map call by capturing array arguments
# with `map_args` and creating a map function with `map_function`.
# Logic from https://github.com/Jutho/Strided.jl/blob/v2.0.4/src/broadcast.jl.

using Base.Broadcast: Broadcasted

const WrappedScalarArgs = Union{AbstractArray{<:Any,0},Ref{<:Any}}

function map_args(bc::Broadcasted, rest...)
  return (map_args(bc.args...)..., map_args(rest...)...)
end
map_args(a::AbstractArray, rest...) = (a, map_args(rest...)...)
map_args(a, rest...) = map_args(rest...)
map_args() = ()

struct MapFunction{F,Args<:Tuple}
  f::F
  args::Args
end
struct Arg end

# construct MapFunction
function map_function(bc::Broadcasted)
  args = map_function_tuple(bc.args)
  return MapFunction(bc.f, args)
end
map_function_tuple(t::Tuple{}) = t
map_function_tuple(t::Tuple) = (map_function(t[1]), map_function_tuple(Base.tail(t))...)
map_function(a::WrappedScalarArgs) = a[]
map_function(a::AbstractArray) = Arg()
map_function(a) = a

# Evaluate MapFunction
(f::MapFunction)(args...) = apply(f, args)[1]
function apply(f::MapFunction, args)
  args, newargs = apply_tuple(f.args, args)
  return f.f(args...), newargs
end
apply(a::Arg, args::Tuple) = args[1], Base.tail(args)
apply(a, args) = a, args
apply_tuple(t::Tuple{}, args) = t, args
function apply_tuple(t::Tuple, args)
  t1, newargs1 = apply(t[1], args)
  ttail, newargs = apply_tuple(Base.tail(t), newargs1)
  return (t1, ttail...), newargs
end
end
