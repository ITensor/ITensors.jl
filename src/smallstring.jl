const IntChar = UInt16
const IntSmallString = UInt256

# XXX: remove smallLength as a global constant, bad for type inference
const smallLength = 16
const SmallStringStorage = SVector{smallLength,IntChar}
const MSmallStringStorage = MVector{smallLength,IntChar}

# Similar types are implemented in various packages:
# https://github.com/JuliaString/ShortStrings.jl
# https://github.com/JuliaComputing/FixedSizeStrings.jl
# https://gist.github.com/SimonDanisch/02e74622e0577f199c1b1a8a65390c24#file-fixedstring-jl
# https://github.com/JuliaStrings/StringViews.jl
# https://discourse.julialang.org/t/way-to-make-sharedarray-over-fixed-length-strings/7082
# https://github.com/djsegal/FixedLengthStrings.jl
# TODO: make this more generic by parametrizing over the length and Char size. Also, store the length of the string.
struct SmallString
  data::SmallStringStorage

  SmallString(sv::SmallStringStorage) = new(sv)

  function SmallString()
    store = SmallStringStorage(ntuple(_ -> IntChar(0), Val(smallLength)))
    return new(store)
  end
end

const Tag = SmallString

data(ss::SmallString) = ss.data

Base.eltype(ss::SmallString) = eltype(data(ss))

function SmallString(str)
  length(str) > smallLength &&
    error("String is too long for SmallString. Maximum length is $smallLength.")
  mstore = MSmallStringStorage(ntuple(_ -> IntChar(0), Val(smallLength)))
  for (n, c) in enumerate(str)
    mstore[n] = IntChar(c)
  end
  return SmallString(SmallStringStorage(mstore))
end

SmallString(s::SmallString) = SmallString(data(s))

Base.getindex(s::SmallString, n::Int) = getindex(s.data, n)

function Base.setindex(s::SmallString, val, n::Int)
  return SmallString(StaticArrays.setindex(s.data, val, n))
end

# TODO: rename to `isempty`
isnull(s::SmallString) = @inbounds s[1] == IntChar(0)

function Base.vcat(s1::SmallString, s2::SmallString)
  v = MSmallStringStorage(ntuple(_ -> IntChar(0), Val(smallLength)))
  n = 1
  while n <= smallLength && s1[n] != IntChar(0)
    v[n] = s1[n]
    n += 1
  end
  N1 = n - 1
  n2 = 1
  while n2 <= smallLength && s2[n2] != IntChar(0)
    v[n] = s2[n2]
    n += 1
    n2 += 1
  end
  return SmallString(SmallStringStorage(v))
end

function SmallString(i::IntSmallString)
  mut_is = MVector{1,IntSmallString}(ntoh(i))
  p = convert(Ptr{SmallStringStorage}, pointer_from_objref(mut_is))
  return SmallString(unsafe_load(p))
end

function cast_to_uint(store)
  mut_store = MSmallStringStorage(store)
  storage_begin = convert(Ptr{IntSmallString}, pointer_from_objref(mut_store))
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
Base.:(==)(s1::SmallString, s2::SmallString) = (s1.data == s2.data)
Base.isless(s1::SmallString, s2::SmallString) = isless(s1.data, s2.data)

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
  len = n - 1
  return String(Char.(s.data[1:len]))
end

function Base.show(io::IO, s::SmallString)
  n = 1
  while n <= smallLength && s[n] != IntChar(0)
    print(io, Char(s[n]))
    n += 1
  end
end

function readcpp(io::IO, ::Type{SmallString}; kwargs...)
  format = get(kwargs, :format, "v3")
  s = SmallString()
  if format == "v3"
    for n in 1:7
      c = read(io, Char)
      s = setindex(s, c, n)
    end
  else
    throw(ArgumentError("read SmallString: format=$format not supported"))
  end
  return s
end
