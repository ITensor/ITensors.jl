
struct TagSet
    tags::Vector{String}
    TagSet(tags::String) = new(sort(split(tags,",")))
    TagSet(tags::Vector{String}) = new(sort(tags))
end

copy(ts::TagSet) = TagSet(ts.tags)

convert(::Type{TagSet},x::String) = TagSet(x)
convert(::Type{TagSet},x::TagSet) = TagSet(x)

==(ts1::TagSet,ts2::TagSet) = (ts1.tags==ts2.tags)

in(tag::String, ts::TagSet) == in(tag, ts.tags)
