replace(collection, replacements::Pair...) = Base.replace(collection, replacements...)
@static if VERSION < v"1.7.0-DEV.15"
  # https://github.com/JuliaLang/julia/pull/38216
  # TODO: Add to `Compat.jl` or delete when we drop Julia 1.6 support.
  # `replace` for Tuples.
  function _replace(f::Base.Callable, t::Tuple, count::Int)
    return if count == 0 || isempty(t)
      t
    else
      x = f(t[1])
      (x, _replace(f, Base.tail(t), count - !==(x, t[1]))...)
    end
  end

  function replace(f::Base.Callable, t::Tuple; count::Integer=typemax(Int))
    return _replace(f, t, Base.check_count(count))
  end

  function _replace(t::Tuple, count::Int, old_new::Tuple{Vararg{Pair}})
    return _replace(t, count) do x
      Base.@_inline_meta
      for o_n in old_new
        isequal(first(o_n), x) && return last(o_n)
      end
      return x
    end
  end

  function replace(t::Tuple, old_new::Pair...; count::Integer=typemax(Int))
    return _replace(t, Base.check_count(count), old_new)
  end
end
