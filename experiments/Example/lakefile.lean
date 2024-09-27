import Lake
open Lake DSL

require proofwidgets from git
  "https://github.com/leanprover-community/ProofWidgets4" @ "ade8c50c8d1172b974738a01447c29bf6f85f7f8"

require aesop from git
  "https://github.com/leanprover-community/aesop.git" @ "v4.10.0-rc1"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.10.0-rc1"

package Example

@[default_target]
lean_lib Example
