from pantograph import Server
from pantograph.server import get_version
from pantograph.expr import Goal, Variable, TacticHave, TacticLet, TacticCalc

def test_version():
    assert get_version() == "0.2.23"

def test_expr_type():
    server = Server()
    t = server.expr_type("forall (n m: Nat), n + m = m + n")
    assert t == "Prop"

def test_goal_start():
    server = Server()
    state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
    assert len(server.to_remove_goal_states) == 0
    assert state0.state_id == 0
    state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a")
    assert state1.state_id == 1
    assert state1.goals == [Goal(
        variables=[Variable(name="a", t="Prop")],
        target="∀ (q : Prop), a ∨ q → q ∨ a",
        name=None,
    )]
    assert str(state1.goals[0]) == "a : Prop\n⊢ ∀ (q : Prop), a ∨ q → q ∨ a"

    del state0
    assert len(server.to_remove_goal_states) == 1
    server.gc()
    assert len(server.to_remove_goal_states) == 0

    state0b = server.goal_start("forall (p: Prop), p -> p")
    del state0b
    assert len(server.to_remove_goal_states) == 1
    server.gc()
    assert len(server.to_remove_goal_states) == 0

def test_automatic_mode():
    server = Server()
    state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
    assert len(server.to_remove_goal_states) == 0
    assert state0.state_id == 0
    state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a b h")
    assert state1.state_id == 1
    assert state1.goals == [Goal(
        variables=[
            Variable(name="a", t="Prop"),
            Variable(name="b", t="Prop"),
            Variable(name="h", t="a ∨ b"),
        ],
        target="b ∨ a",
        name=None,
    )]
    state2 = server.goal_tactic(state1, goal_id=0, tactic="cases h")
    assert state2.goals == [
        Goal(
            variables=[
                Variable(name="a", t="Prop"),
                Variable(name="b", t="Prop"),
                Variable(name="h✝", t="a"),
            ],
            target="b ∨ a",
            name="inl",
        ),
        Goal(
            variables=[
                Variable(name="a", t="Prop"),
                Variable(name="b", t="Prop"),
                Variable(name="h✝", t="b"),
            ],
            target="b ∨ a",
            name="inr",
        ),
    ]
    state3 = server.goal_tactic(state2, goal_id=1, tactic="apply Or.inl")
    state4 = server.goal_tactic(state3, goal_id=0, tactic="assumption")
    assert state4.goals == [
        Goal(
            variables=[
                Variable(name="a", t="Prop"),
                Variable(name="b", t="Prop"),
                Variable(name="h✝", t="a"),
            ],
            target="b ∨ a",
            name="inl",
        )
    ]

def test_have():
    server = Server()
    state0 = server.goal_start("1 + 1 = 2")
    state1 = server.goal_tactic(state0, goal_id=0, tactic=TacticHave(branch="2 = 1 + 1", binder_name="h"))
    assert state1.goals == [
        Goal(
            variables=[],
            target="2 = 1 + 1",
        ),
        Goal(
            variables=[Variable(name="h", t="2 = 1 + 1")],
            target="1 + 1 = 2",
        ),
    ]

def test_let():
    server = Server()
    state0 = server.goal_start("1 + 1 = 2")
    state1 = server.goal_tactic(
        state0, goal_id=0,
        tactic=TacticLet(branch="2 = 1 + 1", binder_name="h"))
    assert state1.goals == [
        Goal(
            variables=[],
            name="h",
            target="2 = 1 + 1",
        ),
        Goal(
            variables=[Variable(name="h", t="2 = 1 + 1", v="?h")],
            target="1 + 1 = 2",
        ),
    ]

def test_conv_calc():
    server = Server(options={"automaticMode": False})
    state0 = server.goal_start("∀ (a b: Nat), (b = 2) -> 1 + a + 1 = a + b")

    variables = [
        Variable(name="a", t="Nat"),
        Variable(name="b", t="Nat"),
        Variable(name="h", t="b = 2"),
    ]
    state1 = server.goal_tactic(state0, goal_id=0, tactic="intro a b h")
    state2 = server.goal_tactic(state1, goal_id=0, tactic=TacticCalc("1 + a + 1 = a + 1 + 1"))
    assert state2.goals == [
        Goal(
            variables,
            target="1 + a + 1 = a + 1 + 1",
            name='calc',
        ),
        Goal(
            variables,
            target="a + 1 + 1 = a + b",
        ),
    ]
    state_c1 = server.goal_conv_begin(state2, goal_id=0)
    state_c2 = server.goal_tactic(state_c1, goal_id=0, tactic="rhs")
    state_c3 = server.goal_tactic(state_c2, goal_id=0, tactic="rw [Nat.add_comm]")
    state_c4 = server.goal_conv_end(state_c3)
    state_c5 = server.goal_tactic(state_c4, goal_id=0, tactic="rfl")
    assert state_c5.is_solved

    state3 = server.goal_tactic(state2, goal_id=1, tactic=TacticCalc("_ = a + 2"))
    state4 = server.goal_tactic(state3, goal_id=0, tactic="rw [Nat.add_assoc]")
    assert state4.is_solved

def test_load_sorry():
    server = Server()
    unit, = server.load_sorry("example (p: Prop): p → p := sorry")
    assert unit.goal_state is not None
    state0 = unit.goal_state
    assert state0.goals == [
        Goal(
            [Variable(name="p", t="Prop")],
            target="p → p",
        ),
    ]
    state1 = server.goal_tactic(state0, goal_id=0, tactic="intro h")
    state2 = server.goal_tactic(state1, goal_id=0, tactic="exact h")
    assert state2.is_solved

def test_env_add_inspect():
    server = Server()
    server.env_add(
        name="mystery",
        t="forall (n: Nat), Nat",
        v="fun (n: Nat) => n + 1",
        is_theorem=False,
    )
    inspect_result = server.env_inspect(name="mystery")
    assert inspect_result['type'] == {'pp': 'Nat → Nat', 'dependentMVars': []}

def test_goal_state_pickling():
    import tempfile
    server = Server()
    state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
    with tempfile.TemporaryDirectory() as td:
        path = td + "/goal-state.pickle"
        server.goal_save(state0, path)
        state0b = server.goal_load(path)
        assert state0b.goals == [
            Goal(
                variables=[],
                target="∀ (p q : Prop), p ∨ q → q ∨ p",
            )
        ]
