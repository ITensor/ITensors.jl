import Base.show

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

mutable struct Index
  id::IDType
  dim::Int
  dir::Arrow
  plev::Int
  tags::TagSet
  Index(id::IDType,
        dim::Integer,
        dir::Arrow,
        plev::Integer,
        tags::TagSet) = new(id,dim,dir,plev,tags)
end

Index() = Index(IDType(0),1,Neither,0,TagSet(""))
Index(dim::Integer,tags="") = Index(rand(IDType),dim,In,0,TagSet(tags))

id(i::Index) = i.id
dim(i::Index) = i.dim
function dim(i1::Index,inds::Index...)
  total_dim = 1
  total_dim *= dim(i1)*dim(inds...)
  return total_dim
end
dim() = 1
dir(i::Index) = i.dir
plev(i::Index) = i.plev
tags(i::Index) = i.tags

==(i1::Index,i2::Index) = (id(i1)==id(i2) && plev(i1)==plev(i2) && tags(i1)==tags(i2))
copy(i::Index) = Index(i.id,i.dim,i.dir,i.plev,copy(i.tags))

dag(i::Index) = Index(id(i),dim(i),-dir(i),plev(i),tags(i))
prime!(i::Index,plinc::Int=1) = (i.plev+=plinc; return i)
prime(i::Index,plinc::Int=1) = (j = copy(i); prime!(j,plinc); return j)
settags(i::Index,tags::String) = Index(id(i),dim(i),dir(i),plev(i),TagSet(tags))
(i::Index)(tags::String) = settags(i,tags)

# Iterating over Index I gives
# integers from 1...dim(I)
start(i::Index) = 1
next(i::Index,n::Int) = (n,n+1)
done(i::Index,n::Int) = (n > dim(i))
colon(n::Int,i::Index) = range(n,dim(i))

function primeString(i::Index)
  pl = plev(i)
  if pl == 0 return ""
  elseif pl > 3 return "'$pl"
  else return "'"^pl
  end
end

function show(io::IO,i::Index) 
    idstr = "$(id(i) % 1000)"
    if length(tags(i)) > 0
      print(io,"($(dim(i)),$(tags(i))|id=$(idstr))$(primeString(i))")
    else
      print(io,"($(dim(i))|id=$(idstr))$(primeString(i))")
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
(i::Index)(n::Int) = IndexVal(i,n)

val(iv::IndexVal) = iv.val
ind(iv::IndexVal) = iv.ind

==(i::Index,iv::IndexVal) = (i==ind(iv))
==(iv::IndexVal,i::Index) = (i==iv)

function show(io::IO,iv::IndexVal) 
  print(io,ind(iv)," = $(val(iv))")
end
