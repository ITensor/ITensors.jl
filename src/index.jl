
const IDType = UInt64

# Arrow direction
@enum Arrow In=-1 Out=1 Neither=0

function -(dir::Arrow)
  if dir==Neither
    error("Cannot reverse direction of Arrow direction 'Neither'")
  else
    return dir==In ? Out : In
  end
end

struct Index
  id::IDType
  dim::Int
  dir::Arrow
  tags::TagSet
  Index(id::IDType,
        dim::Integer,
        dir::Arrow,
        tags::TagSet) = new(id,dim,dir,tags)
end

Index() = Index(IDType(0),1,Neither,TagSet(String[],0))
function Index(dim::Integer,tags::String="0")
  ts = TagSet(tags)
  # By default, an Index has a prime level of 0
  tsplev = plev(ts)==-1 ? 0 : plev(ts)
  Index(rand(IDType),dim,In,TagSet(ts.tags,tsplev))
end

id(i::Index) = i.id

dim() = 1
dim(i::Index) = i.dim

function dim(i1::Index,inds::Index...)
  total_dim = 1
  total_dim *= dim(i1)*dim(inds...)
  return total_dim
end

dir(i::Index) = i.dir
tags(i::Index) = i.tags
plev(i::Index) = plev(tags(i))

==(i1::Index,i2::Index) = (id(i1)==id(i2) && tags(i1)==tags(i2))
copy(i::Index) = Index(i.id,i.dim,i.dir,copy(i.tags))

dag(i::Index) = Index(id(i),dim(i),-dir(i),tags(i))

isdefault(i::Index) = (i==Index())

function prime(i::Index,inc::Int=1)
  return Index(id(i),dim(i),dir(i),prime(tags(i),inc))
end
adjoint(i::Index) = prime(i)
function setprime(i::Index,plev::Int)
  return Index(id(i),dim(i),dir(i),setprime(tags(i),plev))
end
noprime(i::Index) = setprime(i,0)

function addtags(i::Index,
                 ts::AbstractString,
                 tsmatch::String="") 
  return Index(id(i),dim(i),dir(i),addtags(tags(i),TagSet(ts),TagSet(tsmatch)))
end

function removetags(i::Index,
                    ts::AbstractString,
                    tsmatch::String="") 
  return Index(id(i),dim(i),dir(i),removetags(tags(i),TagSet(ts),TagSet(tsmatch)))
end

function settags(i::Index,
                 ts::AbstractString,
                 tsmatch::String="")
  tagsetmatch = TagSet(tsmatch)
  (tagsetmatch≠TagSet() && tagsetmatch≠tags(i)) && return i
  tsnew = TagSet(ts)
  # By default, an Index has a prime level of 0
  tsnewplev = plev(tsnew)==-1 ? 0 : plev(tsnew)
  Index(id(i),dim(i),dir(i),TagSet(tsnew.tags,tsnewplev))
end

(i::Index)(ts::String) = settags(i,ts)

hastags(i::Index,ts::Union{String,TagSet}) = hastags(tags(i),ts)

function replacetags(i::Index,
                     tsold::AbstractString,
                     tsnew::AbstractString,
                     tsmatch::String="") 
  tagsetmatch = TagSet(tsmatch)
  tagsetnew = TagSet(tsnew)
  tagsetold = TagSet(tsold)
  #TODO: Avoid this copy?
  tagsetold∉tags(i) && return copy(i)
  itags = replacetags(tags(i),tagsetold,tagsetnew,tagsetmatch)
  return Index(id(i),dim(i),dir(i),itags)
end

function tags(i::Index,
              ts::AbstractString)
  ts = filter(x -> !isspace(x),ts)
  vts = split(ts,"->")
  length(vts) == 1 && error("Must use -> to replace tags of an Index")
  length(vts) > 2 && error("Can only use a single -> when replacing tags of an Index")
  tsremove,tsadd = vts
  if tsremove==""
    return addtags(i,tsadd)
  #TODO: notation to replace all tags?
  #elseif tsremove=="all"
  #  ires = settags(i,tsadd)
  elseif tsadd==""
    return removetags(i,tsremove)
  else
    return replacetags(i,tsremove,tsadd)
  end
end

# Iterating over Index I gives
# integers from 1...dim(I)
start(i::Index) = 1
next(i::Index,n::Int) = (n,n+1)
done(i::Index,n::Int) = (n > dim(i))
colon(n::Int,i::Index) = range(n,dim(i))

function show(io::IO,
              i::Index) 
  idstr = "$(id(i) % 1000)"
  if length(tags(i)) > 0
    print(io,"($(dim(i))|id=$(idstr)|$(tags(i)))$(primestring(tags(i)))")
  else
    print(io,"($(dim(i))|id=$(idstr))$(primestring(tags(i)))")
  end
end

struct IndexVal
  ind::Index
  val::Int
  function IndexVal(i::Index,n::Int)
    n>dim(i) && throw(ErrorException("Value $n greater than size of Index $i"))
    n<1 && throw(ErrorException("Index value must be >= 1 (was $n)"))
    return new(i,n)
  end
end
getindex(i::Index, j::Int) = IndexVal(i, j)
getindex(i::Index, c::Colon) = [IndexVal(i, j) for j in 1:dim(i)]

(i::Index)(n::Int) = IndexVal(i,n)

val(iv::IndexVal) = iv.val
ind(iv::IndexVal) = iv.ind

==(i::Index,iv::IndexVal) = (i==ind(iv))
==(iv::IndexVal,i::Index) = (i==iv)

plev(iv::IndexVal) = plev(ind(iv))
prime(iv::IndexVal,inc::Integer=1) = IndexVal(prime(ind(iv),inc),val(iv))
adjoint(iv::IndexVal) = IndexVal(adjoint(ind(iv)),val(iv))

show(io::IO,iv::IndexVal) = print(io,ind(iv),"=$(val(iv))")
