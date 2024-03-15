# Labelled object interface.
abstract type LabelledStyle end
struct IsLabelled <: LabelledStyle end
struct NotLabelled <: LabelledStyle end
LabelledStyle(::Type) = NotLabelled()
LabelledStyle(object) = LabelledStyle(typeof(object))
islabelled(::IsLabelled) = true
islabelled(::NotLabelled) = false
islabelled(object) = islabelled(LabelledStyle(object))
label(object) = error("This object does not have a label.")
# TODO: Use `TypeParameterAccessors`.
label_type(::Type) = error("No label type defined.")
label_type(object) = typeof(label(object))
labelled(object, label) = error("Can't add a label to this object.")
# TODO: Turn into a trait function.
function set_label(object, label)
  if islabelled(object)
    object = unlabel(object)
  end
  return labelled(object, label)
end
unlabel(object) = object
unlabel_type(type::Type) = type
unlabel_type(object) = typeof(unlabel(object))

labelled_mul(x, y) = labelled_mul(LabelledStyle(x), x, LabelledStyle(y), y)
labelled_mul(::IsLabelled, x, ::IsLabelled, y) = unlabel(x) * unlabel(y)
labelled_mul(::IsLabelled, x, ::NotLabelled, y) = labelled(unlabel(x) * y, label(x))
labelled_mul(::NotLabelled, x, ::IsLabelled, y) = labelled(x * unlabel(y), label(y))

for (f, labelled_f) in [(:div, :labelled_div), (:/, :labelled_division)]
  @eval begin
    $labelled_f(x, y) = $labelled_f(LabelledStyle(x), x, LabelledStyle(y), y)
    $labelled_f(::IsLabelled, x, ::IsLabelled, y) = $f(unlabel(x), unlabel(y))
    $labelled_f(::IsLabelled, x, ::NotLabelled, y) = labelled($f(unlabel(x), y), label(x))
    $labelled_f(::NotLabelled, x, ::IsLabelled, y) = $f(x, unlabel(y))
  end
end

for f in [:isequal, :isless]
  labelled_f = Symbol(:labelled_, f)
  @eval begin
    $labelled_f(x, y) = $labelled_f(LabelledStyle(x), x, LabelledStyle(y), y)
    $labelled_f(::IsLabelled, x, ::IsLabelled, y) = $f(unlabel(x), unlabel(y))
    $labelled_f(::IsLabelled, x, ::NotLabelled, y) = $f(unlabel(x), y)
    $labelled_f(::NotLabelled, x, ::IsLabelled, y) = $f(x, unlabel(y))
  end
end
