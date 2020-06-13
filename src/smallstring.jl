
const IntChar = UInt16
const IntSmallString = UInt128
const smallLength = 8
const SmallStringStorage = SVector{smallLength,IntChar}
const MSmallStringStorage = MVector{smallLength,IntChar}

struct SmallString
  data::SmallStringStorage

  SmallString(sv::SmallStringStorage) = new(sv)

  function SmallString()
    store = SmallStringStorage(ntuple(_->IntChar(0),Val(smallLength)))
    return new(store)
  end

end

function SmallString(str::AbstractString)
  length(str) > smallLength && error("String is too long for SmallString. Maximum length is $smallLength.")
  mstore = MSmallStringStorage(ntuple(_->IntChar(0),Val(smallLength)))
  for (n,c) in enumerate(str)
    mstore[n] = IntChar(c)
  end
  return SmallString(SmallStringStorage(mstore))
end

SmallString(s::SmallString) = SmallString(s.data)

Base.getindex(s::SmallString, n::Int) = getindex(s.data,n)

function Base.setindex(s::SmallString, val, n::Int)
  return SmallString(StaticArrays.setindex(s.data, val, n))
end

isnull(s::SmallString) = @inbounds s[1] == IntChar(0)

function SmallString(i::IntSmallString)
  mut_is = MVector{1,IntSmallString}(ntoh(i))
  p = convert(Ptr{SmallStringStorage},pointer_from_objref(mut_is))
  return SmallString(unsafe_load(p))
end

function cast_to_uint(store)
  mut_store = MSmallStringStorage(store)
  storage_begin = convert(Ptr{IntSmallString},pointer_from_objref(mut_store))
  return ntoh(unsafe_load(storage_begin))
end

function IntSmallString(s::SmallString)
  return cast_to_uint(s.data)
end

function isint(s::SmallString)::Bool
  ndigits = 1
  while ndigits <= smallLength && s[ndigits] != IntChar(0)
    cur_char = Char(s[ndigits])
    !isdigit(cur_char) && return false
    ndigits += 1
  end
  return true
end

# Here we use the StaticArrays comparison
Base.:(==)(s1::SmallString,s2::SmallString) = (s1.data == s2.data)
Base.isless(s1::SmallString,s2::SmallString) = isless(s1.data,s2.data)

########################################################
# Here are alternative SmallString comparison implementations
#

#Base.isless(a::SmallString,b::SmallString) = cast_to_uint(a) < cast_to_uint(b)
#Base.:(==)(a::SmallString,b::SmallString) = cast_to_uint(a) == cast_to_uint(b)

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
  return String(Char.(s.data[1:len]))
end

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
