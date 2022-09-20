### Should this be with the prime information?
# Maybe better to have a prime.jl file?
#
# not syntax (to prime or tag the compliment
# of the specified indices/pattern)
#

struct Not{T}
  pattern::T
  Not(p::T) where {T} = new{T}(p)
end

"""
not(p)

Represents the compliment of the input
for pattern matching in priming, tagging
and other IndexSet related functions.
"""
function not end

"""
parent(n::Not)

Get the original pattern.
"""
Base.parent(n::Not) = n.pattern
