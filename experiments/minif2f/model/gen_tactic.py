"""
Tactic generation functions for the LLM agent
"""
from pantograph.server import Server, ServerError, TacticFailure
from pantograph.expr import Variable, Goal, TacticCalc
import sglang as sgl
from termcolor import colored
import unittest
from .options import CORE_OPTIONS

LEAN4_INTRO = '''/-- A sequence `u` of real numbers converges to `l` if `∀ ε > 0, ∃ N, ∀ n ≥ N, |u_n - l| ≤ ε`.
This condition will be spelled `seq_limit u l`. -/
def seq_limit (u : ℕ → ℝ) (l : ℝ) : Prop :=
∀ ε > 0, ∃ N, ∀ n ≥ N, |u n - l| ≤ ε

/- In the above definition, note that the `n`-th term of the sequence `u` is denoted
simply by `u n`.

Similarly, in the next definition, `f x` is what we would write `f(x)` on paper.
Also note that implication is denoted by a single arrow (we'll explain why later). -/

/-- A function`f : ℝ → ℝ` is continuous at `x₀` if
`∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| ≤ δ ⇒ |f(x) - f(x₀)| ≤ ε`.
This condition will be spelled `continuous_at f x₀`.-/
def continuous_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| ≤ δ → |f x - f x₀| ≤ ε

/-- Now we claim that if `f` is continuous at `x₀` then it is sequentially continuous
at `x₀`: for any sequence `u` converging to `x₀`, the sequence `f ∘ u` converges
to `f x₀`.  -/
example (f : ℝ → ℝ) (u : ℕ → ℝ) (x₀ : ℝ) (hu : seq_limit u x₀) (hf : continuous_at f x₀) :
  seq_limit (f ∘ u) (f x₀) := by { -- This `by` keyword marks the beginning of the proof
  -- Put your text cursor here and watch the Lean InfoView panel to the right.
  -- Then move your cursor from line to line in the proof while monitoring the Infoview.

  -- Our goal is to prove that, for any positive `ε`, there exists a natural
  -- number `N` such that, for any natural number `n` at least `N`,
  --  `|f(u_n) - f(x₀)|` is at most `ε`.
  unfold seq_limit
  -- Fix a positive number `ε`.
  intros ε hε
  -- By assumption on `f` applied to this positive `ε`, we get a positive `δ`
  -- such that, for all real number `x`, if `|x - x₀| ≤ δ` then `|f(x) - f(x₀)| ≤ ε` (1).
  obtain ⟨δ, δ_pos, Hf⟩ : ∃ δ > 0, ∀ x, |x - x₀| ≤ δ → |f x - f x₀| ≤ ε := hf ε hε
  -- The assumption on `u` applied to this `δ` gives a natural number `N` such that
  -- for every natural number `n`, if `n ≥ N` then `|u_n - x₀| ≤ δ`   (2).
  obtain ⟨N, Hu⟩ : ∃ N, ∀ n ≥ N, |u n - x₀| ≤ δ := hu δ δ_pos
  -- Let's prove `N` is suitable.
  use N
  -- Fix `n` which is at least `N`. Let's prove `|f(u_n) - f(x₀)| ≤ ε`.
  intros n hn
  -- Thanks to (1) applied to `u_n`, it suffices to prove that `|u_n - x₀| ≤ δ`.
  apply Hf
  -- This follows from property (2) and our assumption on `n`.
  exact Hu n hn
  -- This finishes the proof!
  }

/-
Now that this proof is over, you can use the file explorer to the
left of this panel to open the file `Exercises > 01Rewriting.lean`.
-/'''

LEAN4_REWRITE = '''Rewrite tactic tutorial:
example (a b c : Nat) : a + b + c = a + c + b := by
  rw [Nat.add_assoc, Nat.add_comm b, ← Nat.add_assoc]

example (a b c : Nat) : a + b + c = a + c + b := by
  rw [Nat.add_assoc, Nat.add_assoc, Nat.add_comm b]

example (a b c : Nat) : a + b + c = a + c + b := by
  rw [Nat.add_assoc, Nat.add_assoc, Nat.add_comm _ b]

example (f : Nat → Nat) (a : Nat) (h : a + 0 = 0) : f a = f 0 := by
  rw [Nat.add_zero] at h
  rw [h]

def Tuple (α : Type) (n : Nat) :=
  { as : List α // as.length = n }

example (n : Nat) (h : n = 0) (t : Tuple α n) : Tuple α 0 := by
  rw [h] at t
  exact t
'''

PREFIX_CURRENT_GOAL = "The current goal: "

@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))


@sgl.function
def select_tactic(
        s, server, state, goal_id,
        informal_stmt: str = "", informal_proof: str = "",
        feedback_turns: int = 5):

    s += sgl.system("You are an expert in Lean. Choose the next ONE tactic to run given the current proof state and goals.")
    s += sgl.user(LEAN4_REWRITE)
    #s += sgl.user("The current proof state: GoalState(state_id=0, goals=[Goal(variables=[], target='∀ (a b: Nat), (b = 2) -> 1 + a + 1 = a + b', name=None, is_conversion=False)])")
    #s += sgl.assistant("```intros a b h```")
    #s += sgl.user("The current proof state: GoalState(state_id=1, goals=[Goal(variables=[Variable(t='Nat', v=None, name='a'), Variable(t='Nat', v=None, name='b'), Variable(t='b = 2', v=None, name='h')], target='1 + a + 1 = a + b', name=None, is_conversion=False)])")
    #s += sgl.assistant('TacticCalc("1 + a + 1 = a + 1 + 1")')
    s += sgl.user(f"{PREFIX_CURRENT_GOAL}p : Prop\n⊢ ∀ (q: Prop), Or p q -> Or q p")
    s += sgl.assistant('```\nintro q\n```')
    s += sgl.user(f"{PREFIX_CURRENT_GOAL}a b c : Nat\n⊢ a + b + c = a + c + b")
    s += sgl.assistant('```\nrw [Nat.add_assoc, Nat.add_comm b, ← Nat.add_assoc]\n```')
    if informal_stmt and informal_proof:
        s += sgl.user("informal theorem statement: " + informal_stmt)
        s += sgl.user("informal proof: " + informal_proof)
    s += sgl.user(f"{PREFIX_CURRENT_GOAL}{state.goals[goal_id]}")
    for i in range(feedback_turns):
        with s.copy() as tmp:
            tmp += sgl.assistant(sgl.gen("tactic", max_tokens=64))
            # print("==tmp===")
            # print(tmp["tactic"])
            tactic = postprocess_reply(extract_code_from_llm_output(tmp["tactic"]))
        s += sgl.assistant(f"```\n{tactic}\n```")
        success, new_state = apply_tactic(server, state, goal_id, tactic)
        # print("===execute===")
        # print(success, new_state )
        if not success:
            print(colored("[Tactic]", "red"), tactic)
            with s.user():
                s += f"This answer got a Lean compile error:\n{new_state}\n"
                s += "Please try again by taking the Lean compiler feedback."
        else:
            print(colored("[Tactic]", "green"), tactic)
            return tactic, new_state
    return None, None

def apply_tactic(server, state, goal_id, tactic):
    try:
        new_state = server.goal_tactic(state=state, goal_id=goal_id, tactic=tactic)
    except ServerError as e:
        return False, e
    except TacticFailure as e:
        return False, e
    return True, new_state

def extract_code_from_llm_output(reply):
    i = reply.find("```lean")
    if i != -1:
        reply = reply[i + 7:]
        i = reply.find("```")
        reply = reply[:i]
        return reply
    i = reply.find("```")
    if i != -1:
        reply = reply[i + 3:]
        i = reply.find("```")
        reply = reply[:i]
        return reply
    return reply

def postprocess_reply(reply):
    reply = reply.strip()
    if reply and reply[-1] == ",":
        reply = reply[:-1]
    return reply

class TestServerSGL(unittest.TestCase):

    def test_conv_calc_sgl(self):
        n_trails = 5
        sgl.set_default_backend(sgl.OpenAI("gpt-4"))

        server = Server(core_options=CORE_OPTIONS)
        state0 = server.goal_start("∀ (a b: Nat), (b = 2) -> 1 + a + 1 = a + b")
        print("==========state0============")
        print(state0)
        variables = [
            Variable(name="a", t="Nat"),
            Variable(name="b", t="Nat"),
            Variable(name="h", t="b = 2"),
        ]

        state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a b h")
        print("==========state1============")
        print(state1)
        state2 = server.goal_tactic(state1, goal_id=0, tactic=TacticCalc("1 + a + 1 = a + 1 + 1"))
        print("==========state2============")
        print(state2)
        self.assertEqual(state2.goals, [
            Goal(
                variables,
                target="1 + a + 1 = a + 1 + 1",
                name='calc',
            ),
            Goal(
                variables,
                target="a + 1 + 1 = a + b",
            ),
        ])
        state3 = None
        for i in range(n_trails):
            print(f"===============trail {str(i)}============")
            try:
                state = select_tactic.run(server, state2, goal_id = 1)
                state3 = state.ret_value
                for m in state.messages():
                    print(m["role"], ":", m["content"])

                print("\n-- new state --\n", state3)
                break

            except ServerError as e:
                print(f"server error: {e}")
                continue
        state3 = server.goal_tactic(state2, goal_id=1, tactic=TacticCalc("_ = a + 2"))


        print("==========state3============")
        print(state3)
        state4 = None
        for i in range(n_trails):
            print(f"===============trail {str(i)}============")
            try:
                state = select_tactic.run(server, state3, goal_id = 0)
                state4 = state.ret_value
                for m in state.messages():
                    print(m["role"], ":", m["content"])

                print("\n-- new state --\n", state4)
                break

            except ServerError as e:
                print(f"server error: {e}")
                continue

        state4 = server.goal_tactic(state3, goal_id=0, tactic="rw [Nat.add_assoc]")
        print("==========state4============")
        print(state4)
        self.assertTrue(state4.is_solved)


    def test_sglang_openai(self):
        sgl.set_default_backend(sgl.OpenAI("gpt-4"))

        print('\n----- Test sglang ---')
        state = multi_turn_question.run(
            question_1="What is the capital of the United States?",
            question_2="List two local attractions.",
        )

        for m in state.messages():
            print(m["role"], ":", m["content"])

        print("\n-- answer_1 --\n", state["answer_1"])


if __name__ == '__main__':

    unittest.main()
