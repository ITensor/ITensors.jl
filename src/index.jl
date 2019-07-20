export Index,
       IndexVal,
       adjoint,
       dag,
       dim,
       prime,
       noprime,
       addtags,
       settags,
       replacetags,
       removetags,
       hastags,
       id,
       isdefault,
       dir,
       plev,
       tags,
       ind,
       setprime,
       sim,
       Neither,
       In,
       Out,
       val

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

Index() = Index(IDType(0),1,Neither,TagSet("0"))
function Index(dim::Integer,tags="0")
  ts = TagSet(tags)
  # By default, an Index has a prime level of 0
  # A prime level less than 0 is interpreted as the
  # prime level not being set
  plev(ts) < 0 && (ts = setprime(ts,0))
  Index(rand(IDType),dim,Out,ts)
end

id(i::Index) = i.id
dim(i::Index) = i.dim
dir(i::Index) = i.dir
tags(i::Index) = i.tags
plev(i::Index) = plev(tags(i))

"""
==(i1::Index, i1::Index)

Compare indices for equality. First the id's are compared,
then the prime levels are compared, and finally that
tags are compared.
"""
function ==(i1::Index,i2::Index)
  return id(i1) == id(i2) && tags(i1) == tags(i2)
end
copy(i::Index) = Index(id(i),dim(i),dir(i),copy(tags(i)))

sim(i::Index) = Index(rand(IDType),dim(i),dir(i),copy(tags(i)))

dag(i::Index) = Index(id(i),dim(i),-dir(i),tags(i))

isdefault(i::Index) = (i==Index())

hastags(i::Index, ts) = hastags(tags(i),ts)

function settags(i::Index, strts)
  ts = TagSet(strts)
  # By default, an Index has a prime level of 0
  plev(ts) < 0 && (ts = setprime(ts,0))
  Index(id(i),dim(i),dir(i),ts)
end

addtags(i::Index, ts) = settags(i, addtags(tags(i), ts))
removetags(i::Index, ts) = settags(i, removetags(tags(i), ts))
replacetags(i::Index, tsold, tsnew) = settags(i, replacetags(tags(i), tsold, tsnew))

prime(i::Index, plinc = 1) = settags(i, prime(tags(i), plinc))
setprime(i::Index, plev) = settags(i, setprime(tags(i), plev))
noprime(i::Index) = setprime(i, 0)

# To use the notation i' == prime(i)
Base.adjoint(i::Index) = prime(i)

# To use the notation i^5 == prime(i,5)
^(i::Index,pl::Integer) = prime(i,pl)

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
    print(io,"($(dim(i))|id=$(idstr)|$(tagstring(tags(i))))$(primestring(tags(i)))")
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
