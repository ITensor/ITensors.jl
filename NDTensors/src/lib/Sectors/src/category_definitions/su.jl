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

fundamental(::Type{SU{N}}) where {N} = SU{N}(ntuple(i -> Int(i == 1), Val(N)))

adjoint(::Type{SU{N}}) where {N} = SU{N}((ntuple(i -> Int(i == 1) + Int(i < N), Val(N))))

function dimension(s::SU)
  N = groupdim(s)
  l = label(s)
  d = 1
  for k1 in 1:N, k2 in (k1+1):N
    d *= ((k2 - k1) + (l[k1] - l[k2]))//(k2 - k1)
  end
  return Int(d)
end

function dual(s::SU)
  l = label(s)
  nl = ((reverse(cumsum(l[begin:(end - 1)] .- l[(begin + 1):end]))..., 0))
  return typeof(s)(nl)
end

# display SU(N) irrep as a Young tableau with utf8 box char
function Base.show(io::IO, ::MIME"text/plain", s::SU)
  l = label(s)
  if l[1] == 0  # singlet = no box
    println(io, "●")
    return nothing
  end

  println("┌─" * "┬─"^(l[1] - 1) * "┐")
  i = 1
  while l[i + 1] != 0
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

  println(io, "└─", "┴─"^max(0, l[i] - 1), "┘")
  return nothing
end

#
# Specializations for the case SU{2}
# Where irreps specified by dimension "d"
#

dimension(s::SU{2}) = 1 + label(s)[1]

SU{2}(d::Integer) = SU{2}((d - 1, 0))

dual(s::SU{2}) = s

function fusion_rule(s1::SU{2}, s2::SU{2})
  d1, d2 = dimension(s1), dimension(s2)
  return [SU{2}(d) for d in (abs(d1-d2)+1):2:(d1+d2-1)]
end

function Base.show(io::IO, s::SU{2})
  return print(io, "SU{2}(", dimension(s), ")")
end
