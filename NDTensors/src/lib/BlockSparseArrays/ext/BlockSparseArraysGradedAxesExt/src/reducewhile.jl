"""
    reducewhile(f, op, collection, state)

reducewhile(x -> length(x) < 3, vcat, ["a", "b", "c", "d"], 2; init=String[]) ==
  (["b", "c"], 4)
"""
function reducewhile(f, op, collection, state; init)
  prev_result = init
  prev_state = state
  result = prev_result
  while f(result)
    prev_result = result
    prev_state = state
    value_and_state = iterate(collection, state)
    isnothing(value_and_state) && break
    value, state = value_and_state
    result = op(result, value)
  end
  return prev_result, prev_state
end

"""
    groupreducewhile(f, op, collection, ngroups)

groupreducewhile((i, x) -> length(x) ≤ i, vcat, ["a", "b", "c", "d", "e", "f"], 3; init=String[]) ==
  (["a"], ["b", "c"], ["d", "e", "f"])
"""
function groupreducewhile(f, op, collection, ngroups; init)
  state = firstindex(collection)
  return ntuple(ngroups) do group_number
    result, state = reducewhile(x -> f(group_number, x), op, collection, state; init)
    return result
  end
end
