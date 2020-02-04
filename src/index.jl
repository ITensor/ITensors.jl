export Index,
       IndexVal,
       dag,
       prime,
       noprime,
       addtags,
       settags,
       readCpp,
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
An `Index` represents a single tensor index with fixed dimension `dim`. Copies of an Index compare equal unless their 
`tags` are different.

An Index carries a `TagSet`, a set of tags which are small strings that specify properties of the `Index` to help 
distinguish it from other Indices. There is a special tag which is referred to as the integer tag or prime 
level which can be incremented or decremented with special priming functions.

Internally, an `Index` has a fixed `id` number, which is how the ITensor library knows two indices are copies of a 
single original `Index`. `Index` objects must have the same `id`, as well as the `tags` to compare equal.
"""
struct Index{T}
  id::IDType
  space::T
  dir::Arrow
  tags::TagSet

  function Index(id::IDType,dim::T,dir::Arrow,tags::TagSet) where {T}
    # By default, an Index has a prime level of 0
    # A prime level less than 0 is interpreted as the
    # prime level not being set
    !hasplev(tags) && (tags = setprime(tags,0))
    return new{T}(id,dim,dir,tags)
  end

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
  return Index(rand(IDType),dim,Out,ts)
end

"""
    id(i::Index)
Obtain the id of an Index, which is a unique 64 digit integer
"""
id(i::Index) = i.id

"""
    dim(i::Index)
Obtain the dimension of an Index
"""
Tensors.dim(i::Index) = i.space

space(i::Index) = i.space

"""
    dir(i::Index)
Obtain the direction of an Index (In or Out)
"""
dir(i::Index) = i.dir

"""
    tags(i::Index)
Obtain the TagSet of an Index
"""
tags(i::Index) = i.tags

"""
    plev(i::Index)
Obtain the prime level of an Index
"""
plev(i::Index) = plev(tags(i))

# Overload for use in AbstractArray interface for Tensor
#Base.length(i::Index) = dim(i)
#Base.UnitRange(i::Index) = 1:dim(i)
#Base.checkindex(::Type{Bool}, i::Index, val::Int64) = (val ≤ dim(i) && val ≥ 1)

"""
==(i1::Index, i1::Index)

Compare indices for equality. First the id's are compared,
then the prime levels are compared, and finally the
tags are compared.
"""
function Base.:(==)(i1::Index,i2::Index)
  return id(i1) == id(i2) && tags(i1) == tags(i2)
end

"""
    copy(i::Index)
Create a copy of index `i` with identical `id`, `dim`, `dir` and `tags`.
"""
Base.copy(i::Index) = Index(id(i),space(i),dir(i),copy(tags(i)))

"""
    sim(i::Index)
Similar to `copy(i::Index)` except `sim` will produce an `Index` with a new, unique `id` instead of the same `id`.
"""
Tensors.sim(i::Index) = Index(rand(IDType),space(i),dir(i),copy(tags(i)))

"""
    dag(i::Index)
Copy an index `i` and reverse it's direction
"""
dag(i::Index) = Index(id(i),space(i),-dir(i),tags(i))

"""
    isdefault(i::Index)
Check if an `Index` `i` was created with the default options.
"""
isdefault(i::Index) = (i==Index())

"""
    hastags(i::Index,ts)
Check if an `Index` `i` has the provided tags,
which can be a string of comma-separated tags or 
a TagSet object

Example:
    i = Index(2,"Site,SpinHalf,n=3")
    hastags(i,"SpinHalf,Site") == true
    hastags(i,"Link") == false
"""
hastags(i::Index, ts) = hastags(tags(i),ts)

"""
    settags(i::Index,ts)
Return a copy of Index `i` with
tags replaced by the ones given
The `ts` argument can be a comma-separated 
string of tags or a TagSet.

Example:
    i = Index(2,"Site,SpinHalf,n=3")
    hastags(i,"Link") == false
    j = settags(i,"Link,n=4")
    hastags(j,"Link") == true
    hastags(j,"n=4,Link") == true
"""
function settags(i::Index, strts)
  ts = TagSet(strts)
  # By default, an Index has a prime level of 0
  !hasplev(ts) && (ts = setprime(ts,0))
  Index(id(i),space(i),dir(i),ts)
end

"""
    addtags(i::Index,ts)
Return a copy of Index `i` with the
specified tags added to the existing ones.
The `ts` argument can be a comma-separated 
string of tags or a TagSet.
"""
addtags(i::Index, ts) = settags(i, addtags(tags(i), ts))

"""
    removetags(i::Index,ts)
Return a copy of Index `i` with the
specified tags removed. The `ts` argument
can be a comma-separated string of tags or a TagSet.
"""
removetags(i::Index, ts) = settags(i, removetags(tags(i), ts))

"""
    replacetags(i::Index,tsold,tsnew)
If the tag set of `i` contains the tags specified by `tsold`,
replaces these with the tags specified by `tsnew`, preserving
any other tags. The arguments `tsold` and `tsnew` can be
comma-separated strings of tags, or TagSet objects.
"""
replacetags(i::Index, tsold, tsnew) = settags(i, replacetags(tags(i), tsold, tsnew))

"""
    prime(i::Index,plinc = 1)
Return a copy of Index `i` with its
prime level incremented by the amount `plinc`
"""
prime(i::Index, plinc = 1) = settags(i, prime(tags(i), plinc))

"""
    setprime(i::Index,plev)
Return a copy of Index `i` with its
prime level set to `plev`
"""
setprime(i::Index, plev) = settags(i, setprime(tags(i), plev))

"""
    noprime(i::Index)
Return a copy of Index `i` with its
prime level set to zero
"""
noprime(i::Index) = setprime(i, 0)

Base.adjoint(i::Index) = prime(i)

# To use the notation i^5 == prime(i,5)
Base.:^(i::Index,pl::Integer) = prime(i,pl)

# Iterating over Index I gives
# integers from 1...dim(I)
function Base.iterate(i::Index,state::Int=1)
  (state > dim(i)) && return nothing
  return (state,state+1)
end

function Base.show(io::IO,
                   i::Index) 
  idstr = "$(id(i) % 1000)"
  if length(tags(i)) > 0
    print(io,"(dim=$(space(i))|id=$(idstr)|$(tagstring(tags(i))))$(primestring(tags(i)))")
  else
    print(io,"(dim=$(space(i))|id=$(idstr))$(primestring(tags(i)))")
  end
end

struct IndexVal{IndexT}
  ind::IndexT
  val::Int
  function IndexVal(i::IndexT,n::Int) where {IndexT}
    n>dim(i) && throw(ErrorException("Value $n greater than size of Index $i"))
    n<1 && throw(ErrorException("Index value must be >= 1 (was $n)"))
    return new{IndexT}(i,n)
  end
end

IndexVal() = IndexVal(Index(),1)

Base.getindex(i::Index, j::Int) = IndexVal(i, j)
#Base.getindex(i::Index, c::Colon) = [IndexVal(i, j) for j in 1:dim(i)]

(i::Index)(n::Int) = IndexVal(i,n)

val(iv::IndexVal) = iv.val
ind(iv::IndexVal) = iv.ind

Base.:(==)(i::Index,iv::IndexVal) = (i==ind(iv))
Base.:(==)(iv::IndexVal,i::Index) = (i==iv)

plev(iv::IndexVal) = plev(ind(iv))
prime(iv::IndexVal,inc::Integer=1) = IndexVal(prime(ind(iv),inc),val(iv))
Base.adjoint(iv::IndexVal) = IndexVal(adjoint(ind(iv)),val(iv))

Base.show(io::IO,iv::IndexVal) = print(io,ind(iv),"=$(val(iv))")

function readcpp(io::IO,::Type{Index}; kwargs...)
  format = get(kwargs,:format,"v3")
  i = Index()
  if format=="v3"
    tags = readcpp(io,TagSet;kwargs...)
    id = read(io,IDType)
    dim = convert(Int64,read(io,Int32))
    dir_int = read(io,Int32)
    dir = dir_int < 0 ? In : Out
    read(io,8) # Read default IQIndexDat size, 8 bytes
    i = Index(id,dim,dir,tags)
  else
    throw(ArgumentError("read Index: format=$format not supported"))
  end
  return i
end

function HDF5.write(parent::Union{HDF5File,HDF5Group},
                    name::AbstractString,
                    I::Index)
  g = g_create(parent,name)
  attrs(g)["type"] = "Index"
  attrs(g)["version"] = 1
  write(g,"id",id(I))
  write(g,"dim",dim(I))
  write(g,"dir",Int(dir(I)))
  write(g,"tags",tags(I))
end

function HDF5.read(parent::Union{HDF5File,HDF5Group},
                   name::AbstractString,
                   ::Type{Index})
  g = g_open(parent,name)
  if read(attrs(g)["type"]) != "Index"
    error("HDF5 group or file does not contain Index data")
  end
  id = read(g,"id")
  dim = read(g,"dim")
  dir = Arrow(read(g,"dir"))
  tags = read(g,"tags",TagSet)
  return Index(id,dim,dir,tags)
end
