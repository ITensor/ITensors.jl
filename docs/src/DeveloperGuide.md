# Developer Guide


## Keyword Argument Best Practices

Keyword arguments such as `f(x,y; a=1, b=2)` are a powerful Julia feature, but it is easy to
misuse them in library code. Below are the "best practices" for using keyword arguments 
when developing ITensor library code.

A particular challenge how to properly use keyword argument "forwarding" where the notation 
`f(; a, b, kwargs...)` allows any number of keyword arguments to be passed.
If a keyword argument is misspelled, then forwarding keywords with `kwargs...` will
silently allow the misspelling, whereas ideally there would be an error message.

Best practices:

1. **Popping Terminal Keyword Arguments**: 
   When passing keyword arguments downward through a stack of function calls, if a certain keyword
   argument will not be used in any functions further down the stack, then these arguments should
   be *listed explicitly* to remove them from the keyword arguments.

   For example, in a call stack `fA -> fB -> fC` if a keyword argument 
   such as `cutoff` is used in the body of `fB` but not in `fC`, then use the following pattern:

   ```
   function fA(...; kwargs...)
      ...
      fB(...; kwargs...)
      ...
   end

   function fB(...; cutoff, kwargs...) # <- explicitly list cutoff here
      ...
      truncate!(psi; cutoff) # <- fB uses cutoff
      fC(...; kwargs...) # fC does not get passed cutoff
   end

   function fC(...; maxdim, outputlevel) # fC does not use or need the `cutoff` kwarg
     ...
   end
   ```

2. **Leaf Functions Should Not Take `kwargs...`**:
   Functions which are the last in the call stack to take any keyword arguments
   should not take keyword arguments by the `kwargs...` pattern. They should only take an explicit
   list of keyword arguments, so as to ensure that an error is thrown if a keyword argument
   is misspelled or missing (if it has no default value).

   Example: `fC` above is a leaf function and does not have `kwargs...` in its signature.

3. **Use Functions to Set Defaults**:
   Keyword arguments can be made optional by providing default values. To avoid having explicit and
   possibly inconsistent defaults spread all over the library code, use globally defined functions to
   provide these defaults. 

   For example:
   ```
   function sum(A::MPS, B::MPS; cutoff=default_cutoff(), kwargs...)
   ...
   end

   function inner(A::MPS, B::MPS; cutoff=default_cutoff(), kwargs...)
   ...
   end
   ```
   where above the default value for the `cutoff` keyword is provided by a function `default_cutoff()` 
   that is defined for the whole library.

4. **Use Named Tuples to "Tunnel" Keywords to Leaf Functions**:
   This is a more advanced pattern. In certain situations, there might be multiple leaf
   functions depending on the execution pathway of the code or in cases where the leaf function
   is a "callback" passed into the code from the upper-level calling code.

   In such cases, different leaf function implementations may expect different sets of keyword arguments.

   To avoid requiring all leaf functions to take all possible keyword arguments (or to use the `kwargs...`
   pattern as a workaround, breaking rule #2 above), use the following pattern:

   ```
   function fA(callback, psi; callback_args, kwargs...)
     ...
     callback(psi; callback_args...)
     ...
   end

   my_callback(psi; a, b) = ...  # define custom callback function

   # Call fA like this:
   fA(my_callback, psi; callback_args = (; a, b))

   ```

5. **External (non-ITensor) Functions**:
   Though it requires judgment in each case, if the keyword arguments an external 
   (non-ITensor) function accepts are small in number, not expected to change, 
   and known ahead of time, try to list them explicitly if possible (rather than forwarding
   with `kwargs...`). Possible exceptions could be if you want to make use of defaults 
   defined for keyword arguments of an external function.


