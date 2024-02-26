using Test: @inferred, @test

macro test_inferred(ex, kws...)
  @assert ex.head === :call
  # Collect the broken/skip keywords and remove them from the rest of keywords
  @assert all(kw -> kw.head === :(=), kws)
  inferreds = [kw.args[2] for kw in kws if kw.args[1] === :inferred]
  inferred = isempty(inferreds) ? true : only(inferreds)
  wrappeds = [kw.args[2] for kw in kws if kw.args[1] === :wrapped]
  wrapped = isempty(wrappeds) ? false : only(wrappeds)
  kws = filter(kw -> kw.args[1] ∉ (:inferred, :wrapped), kws)
  @assert length(ex.args[2:end]) == 2
  arg1 = ex.args[2]
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
  ex.args[2] = arg1
  return Expr(:macrocall, Symbol("@test"), :(), esc(ex), kws...)
end
