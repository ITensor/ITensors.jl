
const IntChar = UInt8
const IntSmallString = UInt64
const maxTagLength = 8
const SmallStringStorage = SVector{maxTagLength,IntChar}
const MSmallStringStorage = MVector{maxTagLength,IntChar} # Mutable SmallString storage

struct SmallString
  data::SmallStringStorage
  length::Int
  SmallString(sv::SmallStringStorage,l::Int) = new(sv,l)
  function SmallString()
    sv = SmallStringStorage(ntuple(_ -> IntChar(0),Val(length(SmallStringStorage))))
    return new(sv, 0)
  end
end

function SmallString(sv::SmallStringStorage)
  sv[1] == IntChar(0) && return SmallString(sv,0)
  len = 1
  while len < length(SmallStringStorage) && @inbounds sv[len+1] ≠ IntChar(0)
    len += 1
  end
  SmallString(sv,len)
end

Base.length(s::SmallString) = s.length
Base.getindex(s::SmallString,n::Integer) = getindex(s.data,n)
function Base.setindex(s::SmallString,val,n::Integer)
  len = length(s)
  if n > len
    len = n
  end
  return SmallString(setindex(s.data,val,n),len)
end

isNull(s::SmallString) = @inbounds s[1] == IntChar(0)

function StaticArrays.push(s::SmallString,val)
  len = length(s)
  return SmallString(setindex(s.data,val,len+1),len+1)
end

# Cast to SmallString:
function SmallString(i::IntSmallString)
  SmallString(unsafe_load(convert(Ptr{SmallStringStorage},pointer_from_objref(MVector{1,IntSmallString}(ntoh(i))))))
end

# Cast to IntSmallString:
function cast_to_uint64(a)
  return ntoh(unsafe_load(convert(Ptr{IntSmallString},pointer_from_objref(MSmallStringStorage(a)))))
end

#isint(i::IntSmallString) = isint(SmallString(i))

function isint(s::SmallString)
  ndigits = 1
  cur_char = Char(s[ndigits])
  !isdigit(cur_char) && return false
  while ndigits < length(s)
    ndigits += 1
    cur_char = Char(s[ndigits])
    !isdigit(cur_char) && return false
  end
  return true
end

#Base.parse(::Type{Int}, i::IntSmallString) = parse(Int,SmallString(i))

function Base.parse(::Type{Int}, s::SmallString)
  n = length(s)
  int = 0
  for j = 1:n
    int = int*10 + (s[j] - 0x30)  # 0x30 === UInt8('0'), '9'-'0' == 9
  end
  return int
end

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
  res = ""
  for n=1:length(s)
    res *= Char(s[n])
  end
  return res
end

function Base.show(io::IO, s::SmallString)
  for n=1:length(s)
    print(io,Char(s[n]))
  end
end

