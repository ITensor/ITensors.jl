#
# Special unitary group SU{N}
#

struct SU{N,T,M} <: AbstractCategory
  # l is the first row of the
  # Gelfand-Tsetlin (GT) pattern describing
  # an SU(N) irrep
  l::NTuple{M,T}

  function SU{N,T,M}(t::NTuple{M,T}) where {N,T<:Integer,M}
    return N == M + 1 ? new{N,T,M}(t) : error("Invalid tuple length")
  end
end

SU{N}(t::NTuple{M,T}) where {N,T,M} = SU{N,T,M}(t)

SymmetryStyle(::SU) = NonAbelianGroup()

category_label(s::SU) = s.l

groupdim(::SU{N}) where {N} = N

trivial(::Type{<:SU{N}}) where {N} = trivial(SU{N,Int})
trivial(::Type{<:SU{N,T}}) where {N,T} = SU{N}(ntuple(_ -> T(0), Val(N - 1)))

fundamental(::Type{<:SU{N}}) where {N} = fundamental(SU{N,Int})
fundamental(::Type{<:SU{N,T}}) where {N,T} = SU{N}(ntuple(i -> T(i == 1), Val(N - 1)))

adjoint(::Type{<:SU{N}}) where {N} = adjoint(SU{N,Int})
adjoint(::Type{<:SU{N,T}}) where {N,T} = SU{N}((ntuple(i -> T(1 + (i == 1)), Val(N - 1))))

function quantum_dimension(::NonAbelianGroup, s::SU)
  N = groupdim(s)
  l = (category_label(s)..., 0)
  d = 1
  for k1 in 1:N, k2 in (k1 + 1):N
    d *= ((k2 - k1) + (l[k1] - l[k2]))//(k2 - k1)
  end
  return Int(d)
end

function GradedAxes.dual(s::SU)
  l = category_label(s)
  nl = reverse(cumsum((l[begin:(end - 1)] .- l[(begin + 1):end]..., l[end])))
  return typeof(s)(nl)
end

function Base.show(io::IO, s::SU)
  disp = join([string(l) for l in category_label(s)], ", ")
  return print(io, "SU(", groupdim(s), ")[", disp, "]")
end

# display SU(N) irrep as a Young tableau with utf8 box char
function Base.show(io::IO, ::MIME"text/plain", s::SU{N}) where {N}
  l = category_label(s)
  if l[1] == 0  # singlet = no box
    println(io, "●")
    return nothing
  end

  println(io, "┌─" * "┬─"^(l[1] - 1) * "┐")
  i = 1
  while i < N - 1 && l[i + 1] != 0
    println(
      io,
      "├─",
      "┼─"^(l[i + 1] - 1 + (l[i] > l[i + 1])),
      "┴─"^max(0, (l[i] - l[i + 1] - 1)),
      "┤"^(l[i] == l[i + 1]),
      "┘"^(l[i] > l[i + 1]),
    )
    i += 1
  end

  print(io, "└─", "┴─"^max(0, l[i] - 1), "┘")
  return nothing
end

#
# Specializations for the case SU{2}
# Where irreps specified by quantum_dimension "d"
#

# SU2 is an alias for SU{2}
const SU2 = SU{2}

# specific constructor for SU{2} with a half-integer
SU{2}(h::Real) = SU{2}((HalfIntegers.twice(HalfIntegers.HalfInteger(h)),))

quantum_dimension(s::SU{2}) = 1 + Int(category_label(s)[1])

GradedAxes.dual(s::SU{2}) = s

function label_fusion_rule(::Type{<:SU{2}}, s1, s2)
  labels = collect((i,) for i in (abs(s1[1] - s2[1])):2:(s1[1] + s2[1]))
  degen = ones(Int, length(labels))
  return degen, labels
end

# display SU2 using half integers
function Base.show(io::IO, s::SU{2})
  return print(io, "SU(2)[S=", HalfIntegers.half(quantum_dimension(s) - 1), "]")
end

function Base.show(io::IO, ::MIME"text/plain", s::SU{2})
  print(io, "S = ", HalfIntegers.half(quantum_dimension(s) - 1))
  return nothing
end

# Specializations for the case SU{3}
# aimed for testing non-abelian non self-conjugate representations
# TODO replace with generic implementation

function label_fusion_rule(::Type{<:SU{3}}, left, right)
  # Compute SU(3) fusion rules using Littlewood-Richardson rule for Young tableaus.
  # See e.g. Di Francesco, Mathieu and Sénéchal, section 13.5.3.
  if sum(right) > sum(left)  # impose more boxes in left Young tableau
    return label_fusion_rule(SU{3}, right, left)
  end

  if right[1] == 0  # avoid issues with singlet
    return [1], [left]
  end

  left_row1 = left[1]
  left_row2 = left[2]
  right_row1 = right[1]
  right_row2 = right[2]

  irreps = []

  # put a23 boxes on 2nd or 3rd line
  a23max1 = 2 * left_row1  # row2a <= row1a
  a23max2 = right_row1  # a2 + a3 <= total number of a
  a23max = min(a23max1, a23max2)
  for a23 in 0:a23max
    a3min1 = left_row2 + 2 * a23 - left_row1 - right_row1
    a3min2 = left_row2 - left_row1 + a23  # no a below a: row2a <= row1
    a3min = max(0, a3min1, a3min2)
    a3max1 = left_row2  # row3a <= row2a
    a3max2 = a23  # a3 <= a2 + a3
    a3max3 = right_row1 - right_row2  # more a than b, right to left: b2 + b3 <= a1 + a2
    a3max = min(a3max1, a3max2, a3max3)
    for a3 in a3min:a3max
      a2 = a23 - a3
      row1a = left_row1 + right_row1 - a23
      row2a = left_row2 + a23 - a3

      # cannot put any b on 1st line: row1ab = row1a
      b3min1 = row2a + right_row2 - row1a  # row2ab <= row1ab = row1a
      b3min2 = right_row2 + a23 - right_row1
      b3min = max(0, b3min1, b3min2)
      b3max1 = right_row2  # only other.row2 b boxes
      b3max2 = (row2a + right_row2 - a3) ÷ 2  # row3ab >= row2ab
      b3max3 = right_row1 - a3  # more a than b, right to left: b2 <= a1
      b3max4 = row2a - a3  # no b below b: row2a >= row3ab
      b3max = min(b3max1, b3max2, b3max3, b3max4)
      for b3 in b3min:b3max
        b2 = right_row2 - b3
        row2ab = row2a + b2
        row3ab = a3 + b3
        yt = (row1a - row3ab, row2ab - row3ab)

        push!(irreps, yt)
      end
    end
  end

  unique_labels = sort(unique(irreps))
  degen = [count(==(irr), irreps) for irr in unique_labels]

  return degen, unique_labels
end
