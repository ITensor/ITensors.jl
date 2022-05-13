module Ops2

struct Applied{F,Args,Kwargs}
  f::F
  args::Args
  kwargs::Kwargs
  function Applied{F,Args}(f, args::Tuple, kwargs::NamedTuple) where {F,Args}
    return new{F,Args,typeof(kwargs)}(f, args, kwargs)
  end
end
Applied{F,Args}(f, args::Tuple; kwargs...) where {F,Args} = Applied{F,Args}(f, args, NamedTuple(kwargs))
Applied(f, args::Tuple; kwargs...) = Applied{typeof(f),typeof(args)}(f, args; kwargs...)
Applied(f, args...; kwargs...) = Applied(f, args; kwargs...)

struct Op
  which_op
  sites::Tuple
  params::NamedTuple
end
Op(which_op, sites::Tuple; kwargs...) = Op(which_op, sites, NamedTuple(kwargs))
Op(which_op, sites...; kwargs...) = Op(which_op, sites; kwargs...)

end
