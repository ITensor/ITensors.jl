
"""
TagType is a parameterized type which allows
making Index tags into Julia types. One use case
is overloading functions such as `op` which
generates physics operators for indices
with certain tags such as "S=1/2".

To make a TagType, you can use the string
macro notation: `TagType"MyTag"`
"""

struct TagType{T}
end

macro TagType_str(s)
  TagType{Tag(s)}
end

#function val(::TagType{T})::Tag where {T} 
#  return T
#end


struct OpName{T}
end

macro OpName_str(s)
  OpName{Tag(s)}
end

#function val(::OpName{T})::Tag where {T} 
#  return T
#end

# TODO: this should be deprecated in a later 
# version, but leaving it as a fallback in case
# any user code uses the old `op` function 
# pattern
function old_call_op(s::Index,
                     opname::AbstractString;
                     kwargs...)
  use_tag = 0
  nfound = 0
  for n=1:length(tags(s))
    TType = TagType{tags(s)[n]}
    if hasmethod(op,Tuple{TType,Index,AbstractString})
      use_tag = n
      nfound += 1
    end
  end
  if nfound == 0
    throw(ArgumentError("Overload of \"op!\" or \"op\" functions not found for operator name \"$opname\" and Index tags $(tags(s))"))
  elseif nfound > 1
    throw(ArgumentError("Multiple tags from $(tags(s)) overload the function \"op\""))
  end

  ttype = TagType{tags(s)[use_tag]}()
  return op(ttype,s,opname;kwargs...)
end

function _call_op!(s::Index,
                   opname::AbstractString;
                   kwargs...)
  use_tag = 0
  nfound = 0
  for n=1:length(tags(s))
    TType = TagType{tags(s)[n]}
    OpN = OpName{Tag(opname)}
    if hasmethod(op!,Tuple{TType,OpN,ITensor,Index})
      use_tag = n
      nfound += 1
    end
  end
  if nfound == 0
    # Try fallback to older interface:
    return old_call_op(s,opname;kwargs...)

    throw(ArgumentError("Overload of \"op!\" functions not found for operator name \"$opname\" and Index tags $(tags(s))"))
  elseif nfound > 1
    throw(ArgumentError("Multiple tags from $(tags(s)) overload the function \"op!\" for operator name \"$opname\""))
  end

  ttype = TagType{tags(s)[use_tag]}()
  opn = OpName{Tag(opname)}()
  Op = emptyITensor(s',dag(s))
  op!(ttype,opn,Op,s;kwargs...)
  return Op
end

function op(s::Index,
            opname::AbstractString;
            kwargs...)::ITensor

  opname = strip(opname)

  if opname == "Id"
    Op = emptyITensor(dag(s),s')
    for n=1:dim(s)
      Op[dag(s)(n),s'(n)] = 1.0
    end
    return Op
  end

  # Interpret operator names joined by *
  # as acting sequentially on the same site
  starpos = findfirst("*",opname)
  if !isnothing(starpos)
    op1 = opname[1:starpos.start-1]
    op2 = opname[starpos.start+1:end]
    return product(op(s,op1;kwargs...),op(s,op2;kwargs...))
  end

  return _call_op!(s,opname;kwargs...)
end

# Version of `op` taking an array of indices
# and an integer of which Index to use
function op(s::Vector{<:Index},
            opname::AbstractString,
            n::Int;
            kwargs...)::ITensor
  return op(s[n],opname;kwargs...)
end

state(s::Index,n::Integer) = s[n]

function state(s::Index,
               str::String)::IndexVal
  use_tag = 0
  nfound = 0
  for n=1:length(tags(s))
    TType = TagType{tags(s)[n]}
    if hasmethod(state,Tuple{TType,AbstractString})
      use_tag = n
      nfound += 1
    end
  end
  if nfound == 0
    error("Overload of \"state\" function not found for Index tags $(tags(s))")
  elseif nfound > 1
    error("Multiple tags from $(tags(s)) overload the function \"state\"")
  end
  TType = TagType{tags(s)[use_tag]}
  sn = state(TType(),str)
  return s[sn]
end

function state(sset::Vector{<:Index},
               j::Integer,
               st)::IndexVal
  return state(sset[j],st)
end

function siteinds(d::Integer,
                  N::Integer; kwargs...)
  return [Index(d,"Site,n=$n") for n=1:N]
end

function siteinds(str::String,
                  N::Integer; kwargs...)
  TType = TagType{Tag(str)}
  if !hasmethod(siteinds,Tuple{TType,Int})
    error("Overload of \"siteinds\" function not found for tag type \"$str\"")
  end
  return siteinds(TType(),N; kwargs...)
end

function has_fermion_string(s::Index,
                            opname::AbstractString;
                            kwargs...)::Bool
  opname = strip(opname)
  use_tag = 0
  nfound = 0
  for n=1:length(tags(s))
    TType = TagType{tags(s)[n]}
    if hasmethod(has_fermion_string,Tuple{TType,Index,AbstractString})
      use_tag = n
      nfound += 1
    end
  end
  if nfound == 0
    return false
  elseif nfound > 1
    error("Multiple tags from $(tags(s)) overload the function \"has_fermion_string\"")
  end
  TType = TagType{tags(s)[use_tag]}
  return has_fermion_string(TType(),s,opname;kwargs...)
end
