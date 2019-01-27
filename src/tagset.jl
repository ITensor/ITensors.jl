
struct TagSet
    tags::Vector{String}

    TagSet() = new(Vector{String}())

    TagSet(tags::Vector{String}) = new(sort(tags))

    function TagSet(tags::String)
      if tags != ""
        new(sort(split(tags,",")))
      else
        new(Vector{String}())
      end
    end
end

length(T::TagSet) = length(T.tags)
getindex(T::TagSet,n::Int) = T.tags[n]

copy(ts::TagSet) = TagSet(ts.tags)

convert(::Type{TagSet},x::String) = TagSet(x)
convert(::Type{TagSet},x::TagSet) = x

==(ts1::TagSet,ts2::TagSet) = (ts1.tags==ts2.tags)

in(tag::String, ts::TagSet) = in(tag, ts.tags)
#∈(tag::String, ts::TagSet) = in(tag, ts)

function show(io::IO, T::TagSet)
  print(io,"\"")
  lT = length(T)
  if lT > 0
    print(io,T[1])
    for n=2:lT
      print(io,",$(T[n])")
    end
  end
  print(io,"\"")
end
