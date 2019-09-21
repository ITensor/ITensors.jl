export makeTagType,
       op,
       state

function makeTagType(t)
  tag = Tag(t)
  return Val{tag}
end

function _call_op(s::Index,
                  opname::AbstractString;
                  kwargs...)
  use_tag = 0
  nfound = 0
  for n=1:length(tags(s))
    TType = Val{tags(s)[n]}
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
  TType = Val{tags(s)[use_tag]}
  return op(TType(),s,opname;kwargs...)
end

function op(s::Index,
            opname::AbstractString;
            kwargs...)::ITensor
  sP = s'

  opname = strip(opname)

  Op = ITensor(dag(s),sP)
  if opname == "Id"
    for n=1:dim(s)
      Op[dag(s)(n),sP(n)] = 1.0
    end
  else
    # Interpret operator names joined by *
    # as acting on the same site
    starpos = findfirst("*",opname)
    if !isnothing(starpos)
      op1 = opname[1:starpos.start-1]
      op2 = opname[starpos.start+1:end]
      return multSiteOps(op(s,op1;kwargs...),op(s,op2;kwargs...))
    end
    return _call_op(s,opname;kwargs...)
  end
  return Op
end

function op(s::Vector{Index},
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
    TType = Val{tags(s)[n]}
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
  TType = Val{tags(s)[use_tag]}
  sn = state(TType(),str)
  return s[sn]
end

function state(sset::Vector{Index},
               j::Integer,
               st)::IndexVal
  return state(sset[j],st)
end
