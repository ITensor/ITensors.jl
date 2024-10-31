# default behavior: self-dual
dual(r::AbstractUnitRange) = r
nondual(r::AbstractUnitRange) = r
isdual(::AbstractUnitRange) = false

using NDTensors.LabelledNumbers:
  LabelledStyle, IsLabelled, NotLabelled, label, labelled, unlabel

dual(i::LabelledInteger) = labelled(unlabel(i), dual(label(i)))
label_dual(x) = label_dual(LabelledStyle(x), x)
label_dual(::NotLabelled, x) = x
label_dual(::IsLabelled, x) = labelled(unlabel(x), dual(label(x)))

flip(g::AbstractGradedUnitRange) = dual(gradedrange(label_dual.(blocklengths(g))))
