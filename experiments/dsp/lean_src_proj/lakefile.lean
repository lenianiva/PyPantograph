import Lake
open Lake DSL

package «lean_src_proj» {
  -- add any package configuration options here
}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

-- ref: https://github.com/leanprover-community/aesop?tab=readme-ov-file#building
-- ref2: https://chatgpt.com/c/fbb6dde3-46e7-4117-9c02-78e5df1e0df5
-- add aesop pkg as a depedency
require aesop from git
  "https://github.com/JLimperg/aesop"

@[default_target]
lean_lib «LeanSrcProj» {
  -- add any library configuration options here
}
