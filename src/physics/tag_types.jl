export TagType,
       TagType_str,
       op,
       siteinds,
       state


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

function _call_op(s::Index,
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
    error("Overload of \"op\" function not found for Index tags $(tags(s))")
  elseif nfound > 1
    error("Multiple tags from $(tags(s)) overload the function \"op\"")
  end
  TType = TagType{tags(s)[use_tag]}
  return op(TType(),s,opname;kwargs...)
end

function op(s::Index,
            opname::AbstractString;
            kwargs...)::ITensor

  opname = strip(opname)

  if opname == "Id"
    Op = ITensor(dag(s),s')
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
    return matmul(op(s,op1;kwargs...),op(s,op2;kwargs...))
  end

  return _call_op(s,opname;kwargs...)
end

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
