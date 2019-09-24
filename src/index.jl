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
       replacetags!,
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
       val

const IDType = UInt64

"""
   Arrow
`enum` type that can take three values: `In`, `Out`, or `Neither`, representing a directionality
associated with an index, i.e. the index leg is directed into or out of a given tensor
"""
@enum Arrow In=-1 Out=1 Neither=0

"""
    -(dir::Arrow)
Reverse direction of a directed `Arrow`.
"""
function -(dir::Arrow)
  dir==Neither && return Neither #throw(ArgumentError("Cannot reverse direction of Arrow direction 'Neither'"))
  return dir==In ? Out : In
end

"""
An `Index` represents a single tensor index with fixed dimension `dim`. Copies of an Index compare equal unless their 
`tags` are different.

An Index carries a `TagSet`, a set of tags which are small strings that specify properties of the `Index` to help 
distinguish it from other Indices. There is a special tag which is referred to as the integer tag or prime 
level which can be incremented or decremented with special priming functions.

Internally, an `Index` has a fixed `id` number, which is how the ITensor library knows two indices are copies of a 
single original `Index`. `Index` objects must have the same `id`, as well as the `tags` to compare equal.
"""
struct Index
  id::IDType
  dim::Int
  dir::Arrow
  tags::TagSet
end

Index() = Index(IDType(0),1,Neither,TagSet(("",0)))

"""
  Index(dim::Integer, tags=("",0))
Create an `Index` with a unique `id` and a tagset given by `tags`.

Example: create a two dimensional index with tag `l`:
    Index(2, "l")
"""
function Index(dim::Integer,tags=("",0))
  ts = TagSet(tags)
  # By default, an Index has a prime level of 0
  # A prime level less than 0 is interpreted as the
  # prime level not being set
  !hasplev(ts) && (ts = setprime(ts,0))
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

"""
    copy(i::Index)
Create a copy of index `i` with identical `id`, `dim`, `dir` and `tags`.
"""
copy(i::Index) = Index(id(i),dim(i),dir(i),copy(tags(i)))

"""
    sim(i::Index)
Similar to `copy(i::Index)` except `sim` will produce an `Index` with a new, unique `id` instead of the same `id`.
"""
sim(i::Index) = Index(rand(IDType),dim(i),dir(i),copy(tags(i)))

"""
    dag(i::Index)
Copy an index `i` and reverse it's direction
"""
dag(i::Index) = Index(id(i),dim(i),-dir(i),tags(i))

"""
    isdefault(i::Index)
Check if an `Index` `i` was created with the default options.
"""
isdefault(i::Index) = (i==Index())

hastags(i::Index, ts) = hastags(tags(i),ts)

function settags(i::Index, strts)
  ts = TagSet(strts)
  # By default, an Index has a prime level of 0
  !hasplev(ts) && (ts = setprime(ts,0))
  Index(id(i),dim(i),dir(i),ts)
end

addtags(i::Index, ts) = settags(i, addtags(tags(i), ts))
removetags(i::Index, ts) = settags(i, removetags(tags(i), ts))
replacetags(i::Index, tsold, tsnew) = settags(i, replacetags(tags(i), tsold, tsnew))

prime(i::Index, plinc = 1) = settags(i, prime(tags(i), plinc))
setprime(i::Index, plev) = settags(i, setprime(tags(i), plev))
noprime(i::Index) = setprime(i, 0)

Base.adjoint(i::Index) = prime(i)

# To use the notation i^5 == prime(i,5)
^(i::Index,pl::Integer) = prime(i,pl)

# Iterating over Index I gives
# integers from 1...dim(I)
function iterate(i::Index,state::Int=1)
  (state > dim(i)) && return nothing
  return (state,state+1)
end

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

ind(pin::Pair{Index,Int}) = pin.first
val(pin::Pair{Index,Int}) = pin.second
==(i::Index,pin::Pair{Index,Int}) = (i==ind(pin))
==(pin::Pair{Index,Int},i::Index) = (i==pin)

ind(v::AbstractVector{IndexVal}) = ind(v[1])
val(v::AbstractVector{IndexVal}) = val.(v)

plev(iv::IndexVal) = plev(ind(iv))
prime(iv::IndexVal,inc::Integer=1) = IndexVal(prime(ind(iv),inc),val(iv))
adjoint(iv::IndexVal) = IndexVal(adjoint(ind(iv)),val(iv))

show(io::IO,iv::IndexVal) = print(io,ind(iv),"=$(val(iv))")

function IndexVal(ivp::Pair{Index,Int})
  return IndexVal(ivp.first,ivp.second)
end

