###########################
# SiteOp                  # 
###########################

struct SiteOp{O,N}
  name::O
  site::NTuple{N,Int}
  params::NamedTuple
end

SiteOp(op::AbstractArray, site::Tuple) = SiteOp(op, site, NamedTuple())
SiteOp(op::AbstractArray, site::Int...) = SiteOp(op, site)

# Change NamedTuple() to (;) when we drop older Julia versions
SiteOp(name::String, site::Tuple) = SiteOp(name, site, NamedTuple())
SiteOp(name::String, site::Int...) = SiteOp(name, site)
function SiteOp(name::String, site_params::Union{Int,NamedTuple}...)
  return SiteOp(name, Base.front(site_params), last(site_params))
end
SiteOp(name::String, params::NamedTuple, site::Tuple) = SiteOp(name, site, params)
SiteOp(name::String, params::NamedTuple, site::Int...) = SiteOp(name, site, params)

function convert(::Type{SiteOp}, op::Pair{Union{String,AbstractArray},Int})
  return SiteOp(first(op), last(op))
end

name(s::SiteOp) = s.name
site(s::SiteOp) = only(s.site)
sites(s::SiteOp) = s.site
params(s::SiteOp) = s.params

site_or_sites(s::SiteOp{1}) = site(s)
site_or_sites(s::SiteOp) = sites(s)

string_site_or_sites(s::SiteOp{1}) = string(site(s))
string_site_or_sites(s::SiteOp) = string(sites(s))[2:(end - 1)]

show(io::IO, s::SiteOp) = print(io, "\"$(name(s))\"($(string_site_or_sites(s)))")

(s1::SiteOp == s2::SiteOp) = (s1.site == s2.site && s1.name == s2.name)

function isless(s1::SiteOp, s2::SiteOp)
  if site(s1) != site(s2)
    return site(s1) < site(s2)
  end
  return name(s1) < name(s2)
end

###########################
# OpTerm                  # 
###########################

const OpTerm = Vector{SiteOp}

function (o1::OpTerm == o2::OpTerm)
  (length(o1) == length(o2)) || return false
  @inbounds for n in 1:length(o1)
    (o1[n] != o2[n]) && return false
  end
  return true
end

function isless(o1::OpTerm, o2::OpTerm)
  if length(o1) != length(o2)
    return length(o1) < length(o2)
  end
  for n in 1:length(o1)
    if o1[n] != o2[n]
      return (o1[n] < o2[n])
    end
  end
  return false
end

mult(t1::OpTerm, t2::OpTerm) = isempty(t2) ? t1 : vcat(t1, t2)

function isfermionic(t::OpTerm, sites)::Bool
  p = +1
  for op in t
    if has_fermion_string(name(op), sites[site(op)])
      p *= -1
    end
  end
  return (p == -1)
end

###########################
# MPOTerm                 # 
###########################

mutable struct MPOTerm
  coef::ComplexF64
  ops::OpTerm
end
coef(op::MPOTerm) = op.coef
ops(op::MPOTerm) = op.ops

copy(t::MPOTerm) = MPOTerm(coef(t), copy(ops(t)))

function (t1::MPOTerm == t2::MPOTerm)
  return coef(t1) ≈ coef(t2) && ops(t1) == ops(t2)
end

function isless(t1::MPOTerm, t2::MPOTerm)
  if ops(t1) == ops(t2)
    if coef(t1) ≈ coef(t2)
      return false
    else
      ct1 = coef(t1)
      ct2 = coef(t2)
      #"lexicographic" ordering on  complex numbers
      return real(ct1) < real(ct2) || (real(ct1) ≈ real(ct2) && imag(ct1) < imag(ct2))
    end
  end
  return ops(t1) < ops(t2)
end

function MPOTerm(c::Number, op1::Union{String,AbstractArray{<:Number}}, ops_rest...) #where T<:Number
  ops = (op1, ops_rest...)
  starts = findall(x -> (x isa String) || (x isa AbstractArray{<:Number}), ops)
  N = length(starts)
  vop = SiteOp[]
  for n in 1:N
    start = starts[n]
    stop = (n == N) ? lastindex(ops) : (starts[n + 1] - 1)
    vop = [vop; [SiteOp(ops[start:stop]...)]]
  end
  return MPOTerm(c, OpTerm(vop))
end

function MPOTerm(op1::Union{String,AbstractArray}, ops...)
  return MPOTerm(one(Float64), op1, ops...)
end

function MPOTerm(ops::Vector{<:Pair})
  return MPOTerm(Iterators.flatten(ops)...)
end

function Base.show(io::IO, op::MPOTerm)
  c = coef(op)
  if iszero(imag(c))
    print(io, "$(real(c)) ")
  elseif iszero(real(c))
    print(io, "$(imag(c))im ")
  else
    print(io, "($c) ")
  end
  for o in ops(op)
    print(io, "\"$(name(o))\"($(string_site_or_sites(o))) ")
    !isempty(params(o)) && print(io, params(o))
  end
end

(α::Number * op::MPOTerm) = MPOTerm(α * coef(op), ops(op))
(op::MPOTerm * α::Number) = α * op
(op::MPOTerm / α::Number) = MPOTerm(coef(op) / α, ops(op))

############################
## OpSum                 #
############################

"""
An `OpSum` represents a sum of operator
terms.

Often it is used to create matrix
product operator (`MPO`) approximation
of the sum of the terms in the `OpSum` oject.
Each term is a product of local operators
specified by names such as `"Sz"` or `"N"`,
times an optional coefficient which
can be real or complex.

Which local operator names are available
is determined by the function `op`
associated with the `TagType` defined by
special Index tags, such as `"S=1/2"`, `"S=1"`,
`"Fermion"`, and `"Electron"`.
"""
mutable struct OpSum
  data::Vector{MPOTerm}
  OpSum(terms::Vector{MPOTerm}) = new(terms)
end

length(os::OpSum) = length(data(os))
getindex(os::OpSum, I::Int) = data(os)[I]

const AutoMPO = OpSum

"""
    OpSum()
    
Construct an empty `OpSum`.
"""
OpSum() = OpSum(Vector{MPOTerm}())

data(ampo::OpSum) = ampo.data
setdata!(ampo::OpSum, ndata) = (ampo.data = ndata)

push!(ampo::OpSum, term) = push!(data(ampo), term)

Base.:(==)(ampo1::OpSum, ampo2::OpSum) = data(ampo1) == data(ampo2)

Base.copy(ampo::OpSum) = OpSum(copy(data(ampo)))

function Base.deepcopy(ampo::OpSum)
  return OpSum(map(copy, data(ampo)))
end

Base.size(ampo::OpSum) = size(data(ampo))

Base.iterate(os::OpSum, args...) = iterate(data(os), args...)

"""
    add!(ampo::OpSum,
         op1::String, i1::Int)

    add!(ampo::OpSum,
         coef::Number,
         op1::String, i1::Int)

    add!(ampo::OpSum,
         op1::String, i1::Int,
         op2::String, i2::Int,
         ops...)

    add!(ampo::OpSum,
         coef::Number,
         op1::String, i1::Int,
         op2::String, i2::Int,
         ops...)

    +(ampo:OpSum, term::Tuple)

Add a single- or multi-site operator 
term to the OpSum `ampo`. Each operator
is specified by a name (String) and a
site number (Int). The second version
accepts a real or complex coefficient.

The `+` operator version of this function
accepts a tuple with entries either
(String,Int,String,Int,...) or
(Number,String,Int,String,Int,...)
where these tuple values are the same
as valid inputs to the `add!` function.
For inputting a very large number of
terms (tuples) to an OpSum, consider
using the broadcasted operator `.+=`
which avoids reallocating the OpSum
after each addition.

# Examples
```julia
ampo = OpSum()

add!(ampo,"Sz",2,"Sz",3)

ampo += ("Sz",3,"Sz",4)

ampo += (0.5,"S+",4,"S-",5)

ampo .+= (0.5,"S+",5,"S-",6)
```
"""
add!(os::OpSum, t::MPOTerm) = push!(os, t)

add!(os::OpSum, args...) = add!(os, MPOTerm(args...))

"""
    subtract!(ampo::OpSum,
              op1::String, i1::Int,
              op2::String, i2::Int,
              ops...)

    subtract!(ampo::OpSum,
              coef::Number,
              op1::String, i1::Int,
              op2::String, i2::Int,
              ops...)

Subtract a multi-site operator term
from the OpSum `ampo`. Each operator
is specified by a name (String) and a
site number (Int). The second version
accepts a real or complex coefficient.
"""
subtract!(os::OpSum, args...) = add!(os, -MPOTerm(args...))

-(t::MPOTerm) = MPOTerm(-coef(t), ops(t))

function (ampo::OpSum + term::MPOTerm)
  ampo_plus_term = copy(ampo)
  add!(ampo_plus_term, term)
  return ampo_plus_term
end

(ampo::OpSum + term::Tuple) = ampo + MPOTerm(term...)
(ampo::OpSum + term::Vector{<:Pair}) = ampo + MPOTerm(term)

function (ampo::OpSum - term::Tuple)
  ampo_plus_term = copy(ampo)
  subtract!(ampo_plus_term, term...)
  return ampo_plus_term
end

function +(o1::OpSum, o2::OpSum; kwargs...)
  return prune!(sortmergeterms!(OpSum([o1..., o2...])), kwargs...)
end

"""
    prune!(os::OpSum; cutoff = 1e-15)

Remove any MPOTerm with norm(coef) < cutoff
"""
function prune!(os::OpSum; atol=1e-15)
  OS = OpSum()
  for o in os
    norm(ITensors.coef(o)) > atol && push!(OS, o)
  end
  os = OS
  return os
end

#
# ampo .+= ("Sz",1) syntax using broadcasting
#

struct OpSumStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:OpSum}) = OpSumStyle()

struct OpSumAddTermStyle <: Broadcast.BroadcastStyle end

Base.broadcastable(ampo::OpSum) = ampo

Base.BroadcastStyle(::OpSumStyle, ::Broadcast.Style{Tuple}) = OpSumAddTermStyle()

Broadcast.instantiate(bc::Broadcast.Broadcasted{OpSumAddTermStyle}) = bc

function Base.copyto!(ampo, bc::Broadcast.Broadcasted{OpSumAddTermStyle,<:Any,typeof(+)})
  add!(ampo, bc.args[2]...)
  return ampo
end

#
# ampo .-= ("Sz",1) syntax using broadcasting
#

function Base.copyto!(ampo, bc::Broadcast.Broadcasted{OpSumAddTermStyle,<:Any,typeof(-)})
  subtract!(ampo, bc.args[2]...)
  return ampo
end

(α::Number * os::OpSum) = OpSum([α * o for o in os])
(os::OpSum * α::Number) = α * os
(os::OpSum / α::Number) = OpSum([o / α for o in os])

(o1::OpSum - o2::OpSum) = o1 + (-1) * o2

function Base.show(io::IO, ampo::OpSum)
  println(io, "OpSum:")
  for term in data(ampo)
    println(io, "  $term")
  end
end
