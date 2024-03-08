using Test: @inferred, @test

macro test_inferred(ex, kws...)
  @assert ex.head in [:call, :(<:)]
  first_arg = ex.head === :(<:) ? 1 : 2
  @assert length(ex.args[first_arg:end]) == 2
  # Collect the broken/skip keywords and remove them from the rest of keywords
  @assert all(kw -> kw.head === :(=), kws)
  inferreds = [kw.args[2] for kw in kws if kw.args[1] === :inferred]
  inferred = isempty(inferreds) ? true : only(inferreds)
  wrappeds = [kw.args[2] for kw in kws if kw.args[1] === :wrapped]
  wrapped = isempty(wrappeds) ? false : only(wrappeds)
  kws = filter(kw -> kw.args[1] ∉ (:inferred, :wrapped), kws)
  arg1 = ex.args[first_arg]
  arg1 = quote
    if $inferred
      if $wrapped
        @inferred((() -> $arg1)())
      else
        @inferred($arg1)
      end
    else
      $arg1
    end
  end
  ex.args[first_arg] = arg1
  return Expr(:macrocall, Symbol("@test"), :(), esc(ex), kws...)
end
