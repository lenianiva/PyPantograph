-- {
--     "problem": "The perimeter of a rectangle is 24 inches. What is the number of square inches in the maximum possible area for this rectangle?",
--     "level": "Level 3",
--     "type": "Algebra",
--     "solution": "Let one pair of parallel sides have length $x$ and the other pair of parallel sides have length $12-x$. This means that the perimeter of the rectangle is $x+x+12-x+12-x=24$ as the problem states. The area of this rectangle is $12x-x^2$. Completing the square results in $-(x-6)^2+36\\le 36$ since $(x-6)^2\\ge 0$, so the maximum area of $\\boxed{36}$ is obtained when the rectangle is a square of side length 6 inches."
-- }

-- Note: translating this to 2x + 2y = 24, what is xy?
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Ring.Defs
import Mathlib.Tactic.Linarith.Frontend

def valid_perimeter (x y : ℕ) : Prop :=
    2 * x + 2 * y = 24

def area (x y : ℝ) := x * y

theorem rewrite_y_as_x: valid_perimeter x y → y = 12 - x := by
    unfold valid_perimeter
    intro p
    have h0 : 24 = 2 * 12 := by rfl
    have h1 : 2 * x + 2 * y = 2 * (x + y) := by ring
    have h2 : 2 * (x + y) = 2 * 12 → x + y = 12 := by sorry
    have h3 : x + y = 12 → y = 12 - x := by sorry
    rw[h0, h1, h2] at p
