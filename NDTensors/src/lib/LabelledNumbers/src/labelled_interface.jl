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

set_value(x, value) = labelled(value, label(x))

labelled_zero(x) = set_value(x, zero(unlabel(x)))
labelled_one(x) = one(unlabel(x))
labelled_one(type::Type) = one(unlabel_type(type))
labelled_oneunit(x) = set_value(x, one(x))
# TODO: Implement this for types where the label is
# encoded in the type.
labelled_oneunit(type::Type) = error("Not implemented.")

function labelled_binary_op(f, x, y)
  return labelled_binary_op(f, LabelledStyle(x), x, LabelledStyle(y), y)
end
labelled_binary_op(f, ::LabelledStyle, x, ::LabelledStyle, y) = f(unlabel(x), unlabel(y))

# TODO: This is only needed for older Julia versions, like Julia 1.6.
# Delete once we drop support for older Julia versions.
labelled_minus(x) = set_value(x, -unlabel(x))

# TODO: This is only needed for older Julia versions, like Julia 1.6.
# Delete once we drop support for older Julia versions.
labelled_hash(x, h::UInt64) = hash(unlabel(x), h)

for (fname, f) in [
  (:mul, :*),
  (:add, :+),
  (:minus, :-),
  (:division, :/),
  (:div, :÷),
  (:isequal, :isequal),
  (:isless, :isless),
]
  labelled_fname = Symbol(:(labelled_), fname)
  @eval begin
    $labelled_fname(x, y) = labelled_binary_op($f, x, y)
  end
end
