# Injectivization

Reversible programs must preserve enough information to recover their inputs.
When a useful computation is not injective, carry a witness that records the
lost branch or cleanup fact.

`examples/injectivized_max.rev` demonstrates the pattern for `max`: the larger
value is moved into `x`, while `swapped` records whether the original inputs
were exchanged. Running the program backward consumes that witness to restore
the original ordering.
