@eval module $(gensym())
using Test: @test, @testset
using ITensors.ITensorMPS: update_observer!
using Observers: observer

@testset "ITensorsObserversExt" begin
  function iterative_function(niter; observer!, observe_step)
    for n in 1:niter
      if iszero(n % observe_step)
        update_observer!(observer!; iteration=n)
      end
    end
  end

  # Record the iteration
  iteration(; iteration) = iteration

  obs = observer(iteration)
  niter = 100
  iterative_function(niter; (observer!)=obs, observe_step=10)
  
  @test size(obs) == (10,1)
end

end
