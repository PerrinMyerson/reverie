# Refinements

Refinements are runtime-checked predicates attached to typed bindings.

```rev
local n: int where n >= 0 = 0
  skip
delocal n: int where n >= 0 = 0
```

The checker requires refinement expressions to be boolean. The interpreter
checks them when a refined local is created and before it is destroyed, so the
failure points keep source spans and participate in normal diagnostics.
