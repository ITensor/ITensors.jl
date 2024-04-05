#
# Special unitary group SU{N}
#

struct SU{N} <: AbstractCategory
  # l is the first row of the
  # Gelfand-Tsetlin (GT) pattern describing
  # an SU(N) irrep
  #TODO: any way this could be NTuple{N-1,Int} ?
  # not in a natural way
  # see https://discourse.julialang.org/t/addition-to-parameter-of-parametric-type/20059/15
  # and https://github.com/JuliaLang/julia/issues/8472
  # can use https://github.com/vtjnash/ComputedFieldTypes.jl
  # can define SU{N,M} and impose M=N-1 in the constructor
  l::NTuple{N,Int}
end

SymmetryStyle(::SU) = NonAbelianGroup()

category_label(s::SU) = s.l

groupdim(::SU{N}) where {N} = N

trivial(::Type{SU{N}}) where {N} = SU{N}(ntuple(_ -> 0, Val(N)))

fundamental(::Type{SU{N}}) where {N} = SU{N}(ntuple(i -> Int(i == 1), Val(N)))

adjoint(::Type{SU{N}}) where {N} = SU{N}((ntuple(i -> Int(i == 1) + Int(i < N), Val(N))))

function quantum_dimension(::NonAbelianGroup, s::SU)
  N = groupdim(s)
  l = category_label(s)
  d = 1
  for k1 in 1:N, k2 in (k1 + 1):N
    d *= ((k2 - k1) + (l[k1] - l[k2]))//(k2 - k1)
  end
  return Int(d)
end

function GradedAxes.dual(s::SU)
  l = category_label(s)
  nl = ((reverse(cumsum(l[begin:(end - 1)] .- l[(begin + 1):end]))..., 0))
  return typeof(s)(nl)
end

# display SU(N) irrep as a Young tableau with utf8 box char
function Base.show(io::IO, ::MIME"text/plain", s::SU)
  l = category_label(s)
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
# Where irreps specified by quantum_dimension "d"
# TBD remove me?
#

quantum_dimension(s::SU{2}) = 1 + category_label(s)[1]

SU{2}(d::Integer) = SU{2}((d - 1, 0))

GradedAxes.dual(s::SU{2}) = s

function label_fusion_rule(::Type{SU{2}}, s1, s2)
  d1 = s1[1] + 1
  d2 = s2[1] + 1
  labels = collect((abs(d1 - d2) + 1):2:(d1 + d2 - 1))
  degen = ones(Int, length(labels))
  return degen, labels
end

function Base.show(io::IO, s::SU{2})
  return print(io, "SU{2}(", quantum_dimension(s), ")")
end
