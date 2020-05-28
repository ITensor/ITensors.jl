ITensors v0.1.5 Release Notes
==============================

* Fix bug with combiner (uncombining step) when combined Index is not the first one (PR #401)
* Add check to ProjMPO to ensure result of `product` is same order as input tensor (PR #390)

ITensors v0.1.4 Release Notes
==============================

* Add note to docs about requiring Julia 1.4 currently
* Improve error message for non-scalar input to `scalar` (PR #396)
* Export @TagType_str macro (PR #393)
* Fix `productMPS` for complex element type (PR #392)

ITensors v0.1.3 Release Notes
==============================

* Use NDTensors v0.1.3, which fixes a bug when taking the SVD of a complex QN ITensor.

ITensors v0.1.2 Release Notes
==============================

* Add functions `norm(::MPS)`, `norm(::MPO)`, `inner(::MPO, ::MPO)`, as well as `logdot`/`loginner` and `lognorm` for getting the logarithm of the inner product or norm between MPSs/MPOs.

