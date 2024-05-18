import Lake
open Lake DSL

require aesop from git
  "https://github.com/leanprover-community/aesop.git" @ "v4.8.0-rc1"

package «Example» where
  -- add package configuration options here

lean_lib «Example» where
  -- add library configuration options here
