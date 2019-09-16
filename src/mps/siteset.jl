export makeTagType,
       op,
       state

function makeTagType(t)
  tag = Tag(t)
  return Val{tag}
end

function _call_op(s::Index,
                  opname::AbstractString)
  for n=1:length(tags(s))
    TType = Val{tags(s)[n]}
    if hasmethod(op,Tuple{TType,Index,AbstractString})
      return op(TType(),s,opname)
    end
  end
  error("Overload of `op` function not found for Index tags $ts")
end

function op(s::Index,
            opname::AbstractString)::ITensor
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
      return multSiteOps(op(s,op1),op(s,op2))
    end
    return _call_op(s,opname)
  end
  return Op
end

function op(s::Vector{Index},
            opname::AbstractString,
            n::Int)::ITensor
  return op(s[n],opname)
end

function show(io::IO,
              inds::Vector{Index})
  (length(inds) > 0) && print(io,"\n")
  for i=1:length(inds)
    println(io,"  $(inds[i])")
  end
end

state(s::Index,n::Integer) = s[n]

function state(s::Index,
               str::String)::IndexVal
  for n=1:length(tags(s))
    TType = Val{tags(s)[n]}
    if hasmethod(state,Tuple{TType,AbstractString})
      sn = state(TType(),str)
      return s[sn]
    end
  end
  error("Overload of `state` function not found for Index tags $ts")
  return IndexVal()
end

function state(sset::Vector{Index},
               j::Integer,
               st)::IndexVal
  return state(sset[j],st)
end
