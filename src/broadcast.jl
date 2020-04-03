#
# ITensorStyle
#

struct ITensorStyle <: Broadcast.BroadcastStyle end
Broadcast.BroadcastStyle(::Type{<:ITensor}) = ITensorStyle()

Broadcast.broadcastable(T::ITensor) = T

function Base.similar(bc::Broadcast.Broadcasted{ITensorStyle},
                      ::Type{ElT}) where {ElT<:Number}
  A = find_type(ITensor, bc.args)
  return similar(A,ElT)
end

#
# ITensorOpScalarStyle
# Operating with a scalar
#

struct ITensorOpScalarStyle <: Broadcast.BroadcastStyle end

function Base.BroadcastStyle(::ITensorStyle,
                             ::Broadcast.DefaultArrayStyle{0})
  return ITensorOpScalarStyle()
end

Base.BroadcastStyle(::ITensorStyle,
                    ::ITensorOpScalarStyle) = ITensorOpScalarStyle()

Broadcast.instantiate(bc::Broadcast.Broadcasted{ITensorOpScalarStyle}) = bc

function Broadcast.broadcasted(::typeof(Base.literal_pow),
                               ::typeof(^),
                               T::ITensor,
                               x::Val)
  return Broadcast.broadcasted(ITensorOpScalarStyle(),
                               Base.literal_pow,
                               Ref(^), T, Ref(x))
end

function Base.similar(bc::Broadcast.Broadcasted{ITensorOpScalarStyle},
                      ::Type{ElT}) where {ElT<:Number}
  A = find_type(ITensor, bc.args)
  return similar(A,ElT)
end

#
# For arbitrary function chaining f.(g.(h.(x)))
#

function Broadcast.instantiate(bc::Broadcast.Broadcasted{ITensorStyle,
                                                         <:Any,
                                                         <:Function,
                                                         <:Tuple{Broadcast.Broadcasted}})
  return Broadcast.instantiate(Broadcast.broadcasted(bc.f∘bc.args[1].f,bc.args[1].args...))
end

function Broadcast.instantiate(bc::Broadcast.Broadcasted{ITensorStyle,
                                                         <:Any,
                                                         <:Function,
                                                         <:Tuple{Broadcast.Broadcasted{ITensorStyle,
                                                                                       <:Any,
                                                                                       <:Function,
                                                                                       <:Tuple{<:ITensor}}}})
  return Broadcast.broadcasted(bc.f∘bc.args[1].f,
                               bc.args[1].args...)  
end

Broadcast.instantiate(bc::Broadcast.Broadcasted{ITensorStyle}) = bc

#
# Some helper functionality to find certain
# inputs in the argument list
#

"`A = find_type(::Type,As)` returns the first of type Type among the arguments."
find_type(::Type{T},
          args::Tuple) where {T} = find_type(T,
                                             find_type(T, args[1]),
                                             Base.tail(args))
find_type(::Type{T}, x) where {T} = x
find_type(::Type{T}, a::T, rest) where {T} = a
find_type(::Type{T}, ::Any, rest) where {T} = find_type(T, rest)
# If not found, return nothing
find_type(::Type{T}, ::Tuple{}) where {T} = nothing

#
# For B .= α .* A
#

function Base.copyto!(T::ITensor,
                      bc::Broadcast.Broadcasted{ITensorOpScalarStyle,
                                                <:Any,
                                                typeof(*),
                                                <:Tuple{<:Union{<:Number,<:ITensor},
                                                        <:Union{<:Number,<:ITensor}}})
  α = find_type(Number, bc.args)
  A = find_type(ITensor, bc.args)
  if A === T
    scale!(T, α)
  else
    mul!(T, α, A)
  end
  return T
end

#
# For B .= A .^ 2.5
#

function Base.copyto!(R::ITensor,
                      bc::Broadcast.Broadcasted{ITensorOpScalarStyle,
                                                <:Any,
                                                typeof(^)})
  α = find_type(Number, bc.args)
  T = find_type(ITensor, bc.args)
  apply!(R,T,(r,t)->t^α)
  return R
end

#
# For A .= α
#

function Base.copyto!(T::ITensor,
                      bc::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0},
                                                <:Any,
                                                typeof(identity),
                                                <:Tuple{<:Number}})
  fill!(T,bc.args[1])
  return T
end

#
# For B .= A
#

function Base.copyto!(T::ITensor,
                      bc::Broadcast.Broadcasted{ITensorStyle,
                                                <:Any,
                                                typeof(identity),
                                                <:Tuple{<:ITensor}})
  copyto!(T,bc.args[1])
  return T
end

#
# For B .+= A
#

function Base.copyto!(T::ITensor,
                      bc::Broadcast.Broadcasted{ITensorStyle,
                                                <:Any,
                                                typeof(+),
                                                <:Tuple{Vararg{<:ITensor}}})
  if T === bc.args[1]
    add!(T,bc.args[2])
  elseif T === bc.args[2]
    add!(T,bc.args[1])
  else
    error("When adding two ITensors in-place, one must be the same as the output ITensor")
  end
  return T
end

#
# For B .-= A
#

function Base.copyto!(T::ITensor,
                      bc::Broadcast.Broadcasted{ITensorStyle,
                                                <:Any,
                                                typeof(-)})
  if T === bc.args[1]
    add!(T,-1,bc.args[2])
  elseif T === bc.args[2]
    add!(T,-1,bc.args[1])
  else
    error("When adding two ITensors in-place, one must be the same as the output ITensor")
  end
  return T
end

#
# For C .+= α .* A or C .+= α .* A .* B
#

function Base.copyto!(T::ITensor,
                      bc::Broadcast.Broadcasted{ITensorOpScalarStyle,
                                                <:Any,
                                                typeof(+)})
  C = find_type(ITensor, bc.args)
  bc_bc = find_type(Broadcast.Broadcasted, bc.args)
  if T === C
    α = find_type(Number, bc_bc.args)
    A = find_type(ITensor, bc_bc.args)
    if !isnothing(α) && !isnothing(A)
      add!(T, α, A)
    else
      bc_bc_bc = find_type(Broadcast.Broadcasted, bc_bc.args)
      if isnothing(α)
        α = find_type(Number, bc_bc_bc.args)
        B = find_type(ITensor, bc_bc_bc.args)
      else
        A,B = bc_bc_bc.args
      end
      mul!(T, A, B, α, 1)
    end
  else
    error("When adding two ITensors in-place, one must be the same as the output ITensor")
  end
  return T
end

#
# For B .-= α .* A
#

function Base.copyto!(T::ITensor,
                      bc::Broadcast.Broadcasted{ITensorOpScalarStyle,
                                                <:Any,
                                                typeof(-)})
  if T === bc.args[1]
    add!(T,-1,bc.args[2].args...)
  elseif T === bc.args[2]
    add!(T,-1,bc.args[1].args...)
  else
    error("When adding two ITensors in-place, one must be the same as the output ITensor")
  end
  return T
end

#
# For C .= β .* C .+ α .* A or C .= β .* C .+ α .* A .* B
#

function Base.copyto!(T::ITensor,
                      bc::Broadcast.Broadcasted{ITensorOpScalarStyle,
                                                <:Any,
                                                typeof(+),
                                                <:Tuple{Vararg{<:Broadcast.Broadcasted}}})
  bc_α = bc.args[1]
  bc_β = bc.args[2]
  α = find_type(Number, bc_α.args)
  A = find_type(ITensor, bc_α.args)
  β = find_type(Number, bc_β.args)
  C = find_type(ITensor, bc_β.args)
  (T !== A && T !== C) && error("When adding two ITensors in-place, one must be the same as the output ITensor")
  if T === A
    bc_α,bc_β = bc_β,bc_α
    α,β = β,α
    A,C = C,A
  end
  if !isnothing(A) && !isnothing(C) && 
     !isnothing(α) && !isnothing(β)
    add!(T, β, α, A)
  else
    bc_bc_α = find_type(Broadcast.Broadcasted, bc_α.args)
    if isnothing(α)
      α = find_type(Number, bc_bc_α.args)
      B = find_type(ITensor, bc_bc_α.args)
    else
      A,B = bc_bc_α.args
    end
    mul!(T, A, B, α, β)
  end
  return T
end

#
# For B .= A .+ α
#

function Base.copyto!(T::ITensor,
                      bc::Broadcast.Broadcasted{ITensorOpScalarStyle,
                                                <:Any,
                                                typeof(+),
                                                <:Tuple{Vararg{<:Union{<:ITensor,<:Number}}}})
  α = find_type(Number,bc.args)
  A = find_type(ITensor,bc.args)
  tensor(T) .= tensor(A) .+ α
  return T
end

#
# For C .= A .* B
#

function Base.copyto!(T::ITensor,
                      bc::Broadcast.Broadcasted{ITensorStyle,
                                                <:Any,
                                                typeof(*)})
  mul!(T, bc.args[1], bc.args[2])
  return T
end

#
# For C .= α .* A .* B
#

function Base.copyto!(T::ITensor,
                      bc::Broadcast.Broadcasted{ITensorOpScalarStyle,
                                                <:Any,
                                                typeof(*)})
  A = find_type(Union{<:Number,<:ITensor}, bc.args)
  bc_bc = find_type(Broadcast.Broadcasted, bc.args)
  if A isa Number
    mul!(T, bc_bc.args[1], bc_bc.args[2], A)
  else
    mul!(T, A, find_type(ITensor, bc_bc.args), 
               find_type(Number, bc_bc.args))
  end
  return T
end

#
# For B .= f.(A)
#

function Base.copyto!(R::ITensor,
                      bc::Broadcast.Broadcasted{ITensorStyle,
                                                <:Any,
                                                <:Function,
                                                <:Tuple{<:ITensor}})
  apply!(R, bc.args[1], (r,t)->bc.f(t))
  return R
end

#
# For B .+= f.(A)
#

function Base.copyto!(R::ITensor,
                      bc::Broadcast.Broadcasted{ITensorStyle,
                                                <:Any,
                                                typeof(+),
                                                <:Tuple{Vararg{Union{<:ITensor,
                                                                     <:Broadcast.Broadcasted}}}})
  R̃ = find_type(ITensor,bc.args)
  bc2 = find_type(Broadcast.Broadcasted,bc.args)
  if R === R̃
    apply!(R,bc2.args[1],(r,t)->r+bc2.f(t))
  else
    error("In C .= B .+ f.(A), C and B must be the same ITensor")
  end
  return R
end

#
# For B .= f.(B) + g.(A)
#

function Base.copyto!(R::ITensor,
                      bc::Broadcast.Broadcasted{ITensorStyle,
                                                <:Any,
                                                typeof(+),
                                                <:Tuple{Vararg{<:Broadcast.Broadcasted}}})
  bc1 = bc.args[1]
  bc2 = bc.args[2]
  T1 = bc1.args[1]
  f1 = bc1.f
  T2 = bc2.args[1]
  f2 = bc2.f
  if R === T1
    apply!(R,T2,(r,t)->f1(r)+f2(t))
  elseif R === T2
    apply!(R,T1,(r,t)->f2(r)+f1(t))
  else
    error("In C .= f.(B) .+ g.(A), C and B or A must be the same ITensor")
  end
  return R
end

#
# For B .= A .^ 2
#

function Base.copyto!(R::ITensor,
                      bc::Broadcast.Broadcasted{ITensorOpScalarStyle,
                                                <:Any,
                                                typeof(Base.literal_pow)})
  α = find_type(Base.RefValue{<:Val},bc.args).x
  powf = find_type(Base.RefValue{<:Function},bc.args).x
  T = find_type(ITensor,bc.args)
  apply!(R,T,(r,t)->bc.f(^,t,α))
  return R
end

