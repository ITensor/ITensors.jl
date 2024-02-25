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
  for i in 2:length(ex.args)
    arg = ex.args[i]
    if inferred
      if wrapped
        arg = :(@inferred((() -> $arg)()))
      else
        if arg isa Expr && arg.head === :call
          arg = :(@inferred($arg))
        end
      end
    end
    ex.args[i] = arg
  end
  return Expr(:macrocall, Symbol("@test"), :(), esc(ex), kws...)
end
