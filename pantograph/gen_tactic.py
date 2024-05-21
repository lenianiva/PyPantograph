from pantograph.server import Server
from pantograph.expr import Variable, Goal, TacticCalc
import  unittest
import sglang as sgl




@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))


@sgl.function
def select_tactic(s, state):
    s += sgl.system("You are an expert in Lean. Choose the next one tactic to run given the current proof state and goals.")
    s += sgl.user("The current proof state: GoalState(state_id=0, goals=[Goal(variables=[], target='∀ (a b: Nat), (b = 2) -> 1 + a + 1 = a + b', name=None, is_conversion=False)])")
    s += sgl.assistant("```intros a b h```")
    s += sgl.user("The current proof state: GoalState(state_id=1, goals=[Goal(variables=[Variable(t='Nat', v=None, name='a'), Variable(t='Nat', v=None, name='b'), Variable(t='b = 2', v=None, name='h')], target='1 + a + 1 = a + b', name=None, is_conversion=False)])")
    s += sgl.assistant('TacticCalc("1 + a + 1 = a + 1 + 1")')
    s += sgl.user("The current proof state: " + str(state))
    with s.copy() as tmp:
        tmp += sgl.assistant(sgl.gen("tactic", max_tokens=64))
        print("==tmp===")
        print(tmp["tactic"])
        tactic = extract_code_from_llm_output(tmp["tactic"])
    s += sgl.assistant("```"+tactic+"```")
    return tactic


    
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

class TestServerSGL(unittest.TestCase):

    def test_conv_calc_sgl(self):
        sgl.set_default_backend(sgl.OpenAI("gpt-4"))

        server = Server()
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
        state = select_tactic.run(str(state2))
        tactic = state.ret_value
        for m in state.messages():
            print(m["role"], ":", m["content"])

        print("\n-- tactic --\n", tactic)
        
        state3 = server.goal_tactic(state2, goal_id=1, tactic=tactic)
        print("==========state3============")
        print(state3)
        # state4 = server.goal_tactic(state3, goal_id=0, tactic="rw [Nat.add_assoc]")
        # print("==========state4============")
        # print(state4)        
        # self.assertTrue(state4.is_solved)


        # print("==========state2============")
        # print(state2)
        # state_c1 = server.goal_conv_begin(state2, goal_id=0)
        # print("==========state c1============")
        # print(state_c1)
        # state_c2 = server.goal_tactic(state_c1, goal_id=0, tactic="rhs")
        # print("==========state c2============")
        # print(state_c2)
        # state_c3 = server.goal_tactic(state_c2, goal_id=0, tactic="rw [Nat.add_comm]")
        # print("==========state c3============")
        # print(state_c3)
        # state_c4 = server.goal_conv_end(state_c3)
        # print("==========state c4============")
        # print(state_c4)

        # state_c5 = server.goal_tactic(state_c4, goal_id=0, tactic="rfl")
        # print("==========state c5============")
        # print(state_c5)
        # self.assertTrue(state_c5.is_solved)

        # print()


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

