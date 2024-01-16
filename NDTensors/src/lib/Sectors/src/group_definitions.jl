import HalfIntegers: Half

#
# U₁ group (circle group, or particle number, total Sz etc.)
#

struct U1 <: AbstractCategory
  n::Half{Int}
end

label(u::U1) = u.n

dimension(::U1) = 1

trivial(::Type{U1}) = U1(0)

label_fusion_rule(::Type{U1}, n1, n2) = (n1 + n2,)

#
# Cyclic group Zₙ
#

struct Z{N} <: AbstractCategory
  m::Half{Int}
  Z{N}(m) where {N} = new{N}(m % N)
end

label(z::Z) = z.m

modulus(::Z{N}) where {N} = N

dimension(::Z) = 1

trivial(::Type{Z{N}}) where {N} = Z{N}(0)

label_fusion_rule(::Type{Z{N}}, n1, n2) where {N} = ((n1 + n2) % N,)

#
# Special unitary group SU{N}
#

struct SU{N} <: AbstractCategory
  # l is the first row of the 
  # Gelfand-Tsetlin (GT) pattern describing
  # an SU(N) irrep
  #TODO: any way this could be NTuple{N-1,Int} ?
  l::NTuple{N,Int}
end

label(s::SU) = s.l

groupdim(::SU{N}) where {N} = N

trivial(::Type{SU{N}}) where {N} = SU{N}(ntuple(_ -> 0, Val(N)))

function dimension(s::SU)
  N = groupdim(s)
  l = label(s)
  d = 1
  for k1 in 1:N, k2 in (k1 + 1):N
    d *= ((k2 - k1) + (l[k1] - l[k2]))//(k2 - k1)
  end
  return Int(d)
end

#
# Specializations for the case SU{2}
# Where irreps specified by dimension "d"
#

dimension(s::SU{2}) = 1 + label(s)[1]

SU{2}(d::Integer) = SU{2}((d - 1, 0))

function fusion_rule(s1::SU{2}, s2::SU{2})
  d1, d2 = dimension(s1), dimension(s2)
  return [SU{2}(d) for d in (abs(d1 - d2) + 1):2:(d1 + d2 - 1)]
end

function Base.show(io::IO, s::SU{2})
  return print(io, "SU{2}(", dimension(s), ")")
end

#
# Conventional SU2 group
# using "J" labels
#

struct SU2 <: AbstractCategory
  j::Half{Int}
end

label(s::SU2) = s.j

trivial(::Type{SU2}) = SU2(0)

dimension(s::SU2) = 2 * label(s) + 1

label_fusion_rule(::Type{SU2}, j1, j2) = abs(j1 - j2):(j1 + j2)
