export convert,
       setindex,
       readcpp

const IntChar = UInt8
const IntSmallString = UInt64
const smallLength = 8
const SmallStringStorage = SVector{smallLength,IntChar}
const MSmallStringStorage = MVector{smallLength,IntChar} # Mutable SmallString storage

struct SmallString
  data::SmallStringStorage

  SmallString(sv::SmallStringStorage) = new(sv)

  function SmallString()
    store = SmallStringStorage(ntuple(_->IntChar(0),Val(smallLength)))
    return new(store)
  end

end

function SmallString(str::String)
  mstore = MSmallStringStorage(ntuple(_->IntChar(0),Val(smallLength)))
  lastchar = min(length(str),smallLength)
  for n=1:lastchar
    mstore[n] = IntChar(str[n])
  end
  return SmallString(SmallStringStorage(mstore))
end

SmallString(s::SmallString) = SmallString(s.data)

#Base.length(s::SmallString) = s.length

Base.getindex(s::SmallString,n::Integer) = getindex(s.data,n)

function Base.setindex(s::SmallString,val,n::Integer)
  return SmallString(setindex(s.data,val,n))
end

isnull(s::SmallString) = @inbounds s[1] == IntChar(0)

#function StaticArrays.push(s::SmallString,val)
#  newlen = 1
#  while newlen <= smallLength && s[newlen] != IntChar(0)
#    newlen += 1
#  end
#  if newlen > smallLength
#    throw(ErrorException("push: SmallString already at maximum length"))
#  end
#  icval = convert(IntChar,val)
#  return SmallString(setindex(s.data,icval,newlen))
#end

function SmallString(i::IntSmallString)
  mut_is = MVector{1,IntSmallString}(ntoh(i))
  p = convert(Ptr{SmallStringStorage},pointer_from_objref(mut_is))
  return SmallString(unsafe_load(p))
end

function cast_to_uint64(store)
  mut_store = MSmallStringStorage(store)
  storage_begin = convert(Ptr{IntSmallString},pointer_from_objref(mut_store))
  return ntoh(unsafe_load(storage_begin))
end

function IntSmallString(s::SmallString)
  return cast_to_uint64(s.data)
end

#isint(i::IntSmallString) = isint(SmallString(i))

function isint(s::SmallString)::Bool
  ndigits = 1
  while ndigits <= smallLength && s[ndigits] != IntChar(0)
    cur_char = Char(s[ndigits])
    !isdigit(cur_char) && return false
    ndigits += 1
  end
  return true
end

#Base.parse(::Type{Int}, i::IntSmallString) = parse(Int,SmallString(i))

#function Base.parse(::Type{Int}, s::SmallString)
#  n = length(s)
#  int = 0
#  for j = 1:n
#    int = int*10 + parse(Int,Char(s[j]))
#  end
#  return int
#end

# Here we use the StaticArrays comparison
Base.:(==)(s1::SmallString,s2::SmallString) = (s1.data == s2.data)
Base.isless(s1::SmallString,s2::SmallString) = isless(s1.data,s2.data)

########################################################
# Here are alternative SmallString comparison implementations
#

#Base.isless(a::SmallString,b::SmallString) = cast_to_uint64(a) < cast_to_uint64(b)
#Base.:(==)(a::SmallString,b::SmallString) = cast_to_uint64(a) == cast_to_uint64(b)

# Here we use the c-function memcmp (used in Julia string comparison):
#function Base.cmp(a::SmallString, b::SmallString)
#  return ccall(:memcmp, Int32, (Ptr{IntChar}, Ptr{IntChar}, IntChar),
#               a.data, b.data, IntChar(8))
#end
#Base.isless(a::SmallString,b::SmallString) = cmp(a, b) < 0
#Base.:(==)(a::SmallString,b::SmallString) = cmp(a, b) == 0

#######################################################

function Base.String(s::SmallString)
  n = 1
  while n <= smallLength && s[n] != IntChar(0)
    n += 1
  end
  len = n-1
  return String(s.data[1:len])
end

#Base.convert(::Type{String}, s::SmallString) = String(s)

function Base.show(io::IO, s::SmallString)
  n = 1
  while n <= smallLength && s[n] != IntChar(0)
    print(io,Char(s[n]))
    n += 1
  end
end

function readcpp(io::IO,::Type{SmallString}; kwargs...)
  format = get(kwargs,:format,"v3")
  s = SmallString()
  if format=="v3"
    for n=1:7
      c = read(io,Char)
      s = setindex(s,c,n)
    end
  else
    throw(ArgumentError("read SmallString: format=$format not supported"))
  end
  return s
end
