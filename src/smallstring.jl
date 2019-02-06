
struct SmallString
  data::SVector{8,UInt8}
end

function SmallString(s::String)
  N = length(s)
  function g(n::Int)::UInt8
    (n <= N) && return s[n]
    return '\0'
  end
  return SmallString(ntuple(g,8))
end

String(ss::SmallString) = prod(Char.(ss.data))

length(s::SmallString) = length(s.data)
getindex(s::SmallString,n::Integer) = getindex(s.data,n)
 
function cmp(a::SmallString, b::SmallString)
  return ccall(:memcmp, Int32, (Ptr{UInt8}, Ptr{UInt8}, UInt8),
               a.data, b.data, 0x08)
end
isless(a::SmallString,b::SmallString) = cmp(a, b) < 0

function show(io::IO, s::SmallString)
  print(io,'"')
  for n=1:length(s)
    print(io,convert(Char,s[n]))
  end
  print(io,'"')
end

