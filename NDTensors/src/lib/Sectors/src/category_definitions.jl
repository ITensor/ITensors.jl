import HalfIntegers: Half

#
# Group U₁
#

struct U1 <: AbstractGroup
  n::Half{Int}
end

label(u::U1) = u.n

dimension(::U1) = 1

trivial(::Type{U1}) = U1(0)

fusion_rule(::Type{U1},n1,n2) = (n1+n2,)


#
# Cyclic group Zₙ
#

struct Z{N} <: AbstractGroup
  m::Half{Int}
  Z{N}(m) where {N} = new{N}(m%N)
end

label(z::Z) = z.m

modulus(::Z{N}) where {N} = N

dimension(::Z) = 1

trivial(::Type{Z{N}}) where {N} = Z{N}(0)

fusion_rule(::Type{Z{N}},n1,n2) where {N} = ((n1+n2)%N,)

#
# Special unitary group SU{N}
#

struct SU{N} <: AbstractGroup
  # l is the first row of the 
  # Gelfand-Tsetlin (GT) pattern describing
  # an SU(N) irrep
  #TODO: any way this could be NTuple{N-1,Int} ?
  l::NTuple{N,Int} 
end

label(s::SU) = s.l

groupdim(::SU{N}) where {N} = N

trivial(::Type{SU{N}}) where {N} = SU{N}(ntuple(_->0,Val(N)))

function dimension(s::SU)
  N = groupdim(s)
  l = label(s)
  d = 1
  for k1=1:N, k2=k1+1:N
    d *= ((k2-k1)+(l[k1]-l[k2]))//(k2-k1)
  end
  return Int(d)
end

#
# Specializations for the case SU{2}
# Where irreps specified by dimension "d"
#

dimension(s::SU{2}) = 1+label(s)[1]

SU{2}(d::Integer) = SU{2}((d-1,0))

function fusion_rule(s1::SU{2}, s2::SU{2}) 
  d1, d2 = dimension(s1), dimension(s2)
  return [SU{2}(d) for d in (abs(d1 - d2) + 1):2:(d1 + d2 - 1)]
end

function Base.show(io::IO,s::SU{2})
  print(io,"SU{2}(",dimension(s),")")
end


#
# Conventional SU2 group
# using "J" labels
#

struct SU2 <: AbstractGroup
  j::Half{Int}
end

label(s::SU2) = s.j

trivial(::Type{SU2}) = SU2(0)

dimension(s::SU2) = 2*label(s)+1

fusion_rule(::Type{SU2},j1,j2) = abs(j1-j2):(j1+j2)

#
# Quantum group su2ₖ
#

struct su2{k} <: AbstractCategory
  j::Half{Int}
end

label(s::su2) = s.j

level(s::su2{k}) where {k} = k

trivial(::Type{su2{k}}) where {k} = su2{k}(0)

function fusion_rule(::Type{su2{k}},j1,j2) where {k}
  return abs(j1 - j2):min(k - j1 - j2, j1 + j2)
end


#
# Fibonacci category
#
# (same fusion rules as subcategory {0,1} of su2{3})
#

struct Fib <: AbstractCategory
  l::Int
end

function Fib(s::AbstractString) 
  if s=="1"
    return Fib(0)
  elseif s=="τ"
    return Fib(1)
  end
  error("Unrecognized input \"$s\" to Fib constructor")
end

label(f::Fib) = f.l

dimension(f::Fib) = (label(f)==0 ? 1 : (1+√5)/2)

# Fusion rules identical to su2₃
fusion_rule(::Type{Fib},l1,l2) = fusion_rule(su2{3},l1,l2)

trivial(::Type{Fib}) = Fib(0)

label_to_str(f::Fib) = label(f)==0 ? "1" : "τ"

function Base.show(io::IO,f::Fib)
  print(io,"Fib(",label_to_str(f),")")
end

#
# Ising category
#
# (same fusion rules as su2{2})
#

struct Ising <: AbstractCategory
  l::Half{Int}
end

function Ising(s::AbstractString) 
  for (a, v) in enumerate(("1", "σ", "ψ"))
    (v == s) && return Ising((a - 1)//2)
  end
  error("Unrecognized input \"$s\" to Ising constructor")
end

label(i::Ising) = i.l

dimension(i::Ising) = (label(i)==1//2) ? √2 : 1

# Fusion rules identical to su2₂
fusion_rule(::Type{Ising},l1,l2) = fusion_rule(su2{2},l1,l2)

trivial(::Type{Ising}) = Ising(0)

label_to_str(i::Ising) = ("1", "σ", "ψ")[Int(2 * label(i) + 1)]

function Base.show(io::IO,f::Ising)
  print(io,"Ising(",label_to_str(f),")")
end

