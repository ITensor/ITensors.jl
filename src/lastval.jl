
struct LastVal{F}
  f::F
end

LastVal() = LastVal(identity)

# TODO: make these definition work for notation
# A[1, end-1]

(l::LastVal + n::Integer) = LastVal(x -> l.f(x) + n)
(n::Integer + l::LastVal) = LastVal(x -> n + l.f(x))
(l::LastVal - n::Integer) = LastVal(x -> l.f(x) - n)
(n::Integer - l::LastVal) = LastVal(x -> n - l.f(x))
(l::LastVal * n::Integer) = LastVal(x -> l.f(x) * n)
(n::Integer * l::LastVal) = LastVal(x -> n * l.f(x))
(-l::LastVal) = LastVal(x -> -l.f(x))
^(l::LastVal, n::Integer) = LastVal(x -> l.f(x)^n)

lastval_to_int(n::Int, l::LastVal) = l.f(n)
lastval_to_int(::Int, n::Int) = n
lastval_to_int(dimsT::Tuple, I::Tuple) = lastval_to_int.(dimsT, I)
lastval_to_int(A::Tensor, I::Tuple) = lastval_to_int(size(A), I)
