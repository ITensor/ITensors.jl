
function getchar(s::String,N::Int,n::Int)
  (n <= N) && return UInt8(s[n])
  return UInt8('\0')
end

const SmallSize = 8

struct SmallString
  data::SVector{SmallSize,UInt8}
  function SmallString(s::String="")
    N = length(s)
    return new(@SVector [getchar(s,N,i) for i = 1:SmallSize])
  end
end

String(ss::SmallString) = prod(Char.(ss.data))

length(s::SmallString) = length(s.data)
getindex(s::SmallString,n::Integer) = getindex(s.data,n)
 
function cmp(a::SmallString, b::SmallString)
  return ccall(:memcmp, Int32, (Ptr{UInt8}, Ptr{UInt8}, UInt8),
               a.data, b.data, convert(UInt8,SmallSize))
end
isless(a::SmallString,b::SmallString) = cmp(a, b) < 0

function show(io::IO, s::SmallString)
  print(io,'"')
  for n=1:length(s)
    print(io,convert(Char,s[n]))
  end
  print(io,'"')
end

