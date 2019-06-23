export SmallArray,_def

#
# Ideas to improve:
#  - make immutable by switching currently mutating functions
#    to just create a new SmallArray instance and return that
#  - just make the storage Union{Tuple,Vector} ?
#  - put an "internal reference" that points to tstore or vstore,
#    may be same as using Union
#  - any value or possibility of making immutable?
#

_def(t::Type{Int64}) = 0
_def(t::Type{Index}) = Index()
  
#function default(t::T)::T where {T}
#  return T()
#end

mutable struct SmallArray{T} <: AbstractVector{T}
  length::Int
  tstore::Tuple{T,T,T,T}
  vstore::Union{Bool,Vector{T}}

  SmallArray{T}(t1::T) where {T} = new(1,(t1,_def(T),_def(T),_def(T)),false)

  SmallArray{T}(t1::T,t2::T) where {T} = new{T}(2,(t1,t2,_def(T),_def(T)),false)

  SmallArray{T}(t1::T,t2::T,t3::T) where {T} = new{T}(3,(t1,t2,t3,_def(T)),false)

  SmallArray{T}(t1::T,t2::T,t3::T,t4::T) where {T} = new{T}(4,(t1,t2,t3,t4),false)

  function SmallArray{T}(t1::T,ts::T...) where {T} 
    vec = T[t1,ts...]
    return new{T}(length(vec),(_def(T),_def(T),_def(T),_def(T)),vec)
  end

  function SmallArray{T}(vt::Vector{T}) where {T}
    if length(vt) > 4
      return new{T}(length(vt),(_def(T),_def(T),_def(T),_def(T)),vt)
    end
    return new{T}(length(vt),tuple(vt...,ntuple(i->_def(T),4-length(vt))...),false)
  end

  #function SmallArray{T}(tt::NTuple{N,T}) where {N,T}
  #  if N > 4
  #    return new{T}(N,(0,0,0,0),Vector{T}(tt))
  #  end
  #  return new{T}(N,tt,false)
  #end

end

Base.convert(::Type{SmallArray{T}},v::Vector{T}) where {T} = SmallArray{T}(v)

#Broadcast

function Base.push!(sa::SmallArray{T},t::T) where T
  len = length(sa)
  sa.length += 1
  if len > 4
    Base.push!(sa.vstore,t)
  elseif len == 4
    sa.vstore = Vector{T}(undef,sa.length)
    for n=1:len
      sa.vstore[n] = sa.tstore[n]
    end
    sa.vstore[sa.length] = t
    sa.tstore = (_def(T),_def(T),_def(T),_def(T))
  else
    sa.tstore = setindex(sa.tstore,t,sa.length)
  end
end

Base.length(sa::SmallArray) = sa.length

function Base.getindex(sa::SmallArray,n::Int)
  if length(sa) > 4
    return sa.vstore[n]
  end
  return sa.tstore[n]
end

function Base.setindex!(sa::SmallArray{T},t::T,n::Int) where {T}
  if length(sa) > 4
    Base.setindex!(sa.vstore,t,n)
  end
  sa.tstore = setindex(sa.tstore,t,n)
end

Base.size(sa::SmallArray) = sa.length

function Base.iterate(sa::SmallArray,state::Int=1)
  if length(sa) > 4
    return Base.iterate(sa.vstore,state)
  end
  return Base.iterate(sa.tstore,state)
end
