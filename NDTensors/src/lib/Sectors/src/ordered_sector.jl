struct OrderedSector <: AbstractCategory
  data::Vector{AbstractCategory}
end

data(os::OrderedSector) = os.data

Base.length(os::OrderedSector) = length(data(os))
Base.getindex(os::OrderedSector, n) = getindex(data(os), n)
Base.isempty(os::OrderedSector) = isempty(data(os))
Base.:(==)(o1::OrderedSector, o2::OrderedSector) = (data(o1) == data(o2))

×(c1::AbstractCategory, c2::AbstractCategory) = OrderedSector([c1, c2])

function Base.show(io::IO, os::OrderedSector)
  isempty(os) && return nothing
  print(io, "(")
  symbol = ""
  for l in data(os)
    print(io, symbol, l)
    symbol = " × "
  end
  return print(io, ")")
end

# Helper function for ⊗
function replace(o::OrderedSector, n::Int, val)
  d = copy(data(o))
  d[n] = val
  return OrderedSector(d)
end

function ⊗(o1::OrderedSector, o2::OrderedSector)
  N = length(o1)
  @assert length(o2) == N
  os = [o1]
  for n in 1:N
    os = [replace(o, n, f) for f in ⊗(o1[n], o2[n]) for o in os]
  end
  return os
end
