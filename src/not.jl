export not

#
# not syntax (to prime or tag the compliment
# of the specified indices/pattern)
#

struct Not{T}
  pattern::T
  Not(p::T) where {T} = new{T}(p)
end
not(p) = Not(p)
Base.parent(n::Not) = n.pattern

