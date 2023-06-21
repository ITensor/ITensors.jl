
#
# Block
#

struct Block{N}
  data::NTuple{N,UInt}
  hash::UInt
  function Block{N}(data::NTuple{N,UInt}) where {N}
    h = _hash(data)
    return new{N}(data, h)
  end
  function Block{0}(::Tuple{})
    h = _hash(())
    return new{0}((), h)
  end
end

#
# Constructors
#

Block{N}(t::Tuple{Vararg{Any,N}}) where {N} = Block{N}(UInt.(t))

Block{N}(I::CartesianIndex{N}) where {N} = Block{N}(I.I)

Block{N}(v::MVector{N}) where {N} = Block{N}(Tuple(v))

Block{N}(v::SVector{N}) where {N} = Block{N}(Tuple(v))

Block(b::Block) = b

Block(I::CartesianIndex{N}) where {N} = Block{N}(I)

Block(v::MVector{N}) where {N} = Block{N}(v)

Block(v::SVector{N}) where {N} = Block{N}(v)

Block(t::NTuple{N,UInt}) where {N} = Block{N}(t)

Block(t::Tuple{Vararg{Any,N}}) where {N} = Block{N}(t)

Block(::Tuple{}) = Block{0}(())

Block(I::Union{Integer,Block{1}}...) = Block(I)

#
# Conversions
#

CartesianIndex(b::Block) = CartesianIndex(Tuple(b))

Tuple(b::Block{N}) where {N} = NTuple{N,UInt}(b.data)

convert(::Type{Block}, I::CartesianIndex{N}) where {N} = Block{N}(I.I)

convert(::Type{Block{N}}, I::CartesianIndex{N}) where {N} = Block{N}(I.I)

convert(::Type{Block}, t::Tuple) = Block(t)

convert(::Type{Block{N}}, t::Tuple) where {N} = Block{N}(t)

(::Type{IntT})(b::Block{1}) where {IntT<:Integer} = IntT(only(b))

#
# Getting and setting fields
#

gethash(b::Block) = b.hash[]

sethash!(b::Block, h::UInt) = (b.hash[] = h; return b)

#
# Basic functions
#

length(::Block{N}) where {N} = N

isless(b1::Block, b2::Block) = isless(Tuple(b1), Tuple(b2))

iterate(b::Block, args...) = iterate(b.data, args...)

@propagate_inbounds function getindex(b::Block, i::Integer)
  return b.data[i]
end

@propagate_inbounds function setindex(b::Block{N}, val, i::Integer) where {N}
  return Block{N}(setindex(b.data, UInt(val), i))
end

ValLength(::Type{<:Block{N}}) where {N} = Val{N}

deleteat(b::Block, pos) = Block(deleteat(Tuple(b), pos))

insertafter(b::Block, val, pos) = Block(insertafter(Tuple(b), UInt.(val), pos))

getindices(b::Block, I) = getindices(Tuple(b), I)

#
# checkbounds
#

# XXX: define this properly
checkbounds(::Tensor, ::Block) = nothing

#
# Hashing
#

# Borrowed from:
# https://github.com/JuliaLang/julia/issues/37073
# This is the same as Julia's Base tuple hash, but is
# a bit faster.
_hash(t::Tuple) = _hash(t, zero(UInt))
_hash(::Tuple{}, h::UInt) = h + Base.tuplehash_seed
@generated function _hash(b::NTuple{N}, h::UInt) where {N}
  quote
    out = h + Base.tuplehash_seed
    @nexprs $N i -> out = hash(b[$N - i + 1], out)
  end
end

if VERSION < v"1.7.0-DEV.933"
  # Stop inlining after some number of arguments to avoid code blowup
  function _hash(t::Base.Any16, h::UInt)
    out = h + Base.tuplehash_seed
    for i in length(t):-1:1
      out = hash(t[i], out)
    end
    return out
  end
else
  # Stop inlining after some number of arguments to avoid code blowup
  function _hash(t::Base.Any32, h::UInt)
    out = h + Base.tuplehash_seed
    for i in length(t):-1:1
      out = hash(t[i], out)
    end
    return out
  end
end

hash(b::Block) = UInt(b.hash)
hash(b::Block, h::UInt) = h + hash(b)

#
# Custom NTuple{N, Int} hashes
# These are faster, but have a lot of collisions
#

# Borrowed from:
# https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
# This seems to have a lot of clashes
#function hash(b::Block, seed::UInt)
#  h = UInt(0x9e3779b9)
#  for n in b
#    seed ⊻= n + h + (seed << 6) + (seed >> 2)
#  end
#  return seed
#end

# Borrowed from:
# http://www.docjar.com/html/api/java/util/Arrays.java.html
# Could also consider uring the CPython tuple hash:
# https://github.com/python/cpython/blob/0430dfac629b4eb0e899a09b899a494aa92145f6/Objects/tupleobject.c#L406
#function hash(b::Block, h::UInt)
#  h += Base.tuplehash_seed
#  for n in b
#    h = 31 * h + n ⊻ (n >> 32)
#  end
#  return h
#end

#
# Printing for Block type
#

show(io::IO, mime::MIME"text/plain", b::Block) = print(io, "Block$(Int.(Tuple(b)))")

show(io::IO, b::Block) = show(io, MIME("text/plain"), b)
