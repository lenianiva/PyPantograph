import Lake
open Lake DSL

-- require aesop from git
--   "https://github.com/leanprover-community/aesop.git" @ "v4.8.0-rc1"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.8.0-rc1"

package Example

@[default_target]
lean_lib Example
