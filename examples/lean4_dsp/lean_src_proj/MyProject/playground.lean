-- import data.real.basic
import Mathlib.Data.Real.Basic

def α : Type := Nat
#check α

-- Cartesian product of Nat and String.
def myTuple : Prod Nat String := (42, "Answer")

#check Prod
#check Prop

def maximum (n : Nat) (k : Nat) : Nat :=
  if n < k then
    k
  else n

#check maximum
#eval maximum 3 4

-- def c : Real := 3
-- def f (x : R) : R :=
-- if h : x = 3 then
--   have h : x = 3 := h
--   x
-- else
--   2 * x

-- open Real

def f (x : Real) : Real := x
#check f
#eval f 3

noncomputable def g (x : ℝ) : ℝ := 1 / x
#check g
#eval g 3

def n : Nat := 3
#eval n
def frac_val : Rat := 1/2
#eval frac_val
def frac_val' : Rat := 0.5
#eval frac_val'

def real_val : ℝ := 0.5
#eval real_val

--
