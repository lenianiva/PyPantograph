examples_for_dsp_draft_prompt_original = [
    {"tag": "aimeI_2000_p7", "category": "algebra", "metadata": {}, "prompt": "Informal:\n(*### Problem\n\nSuppose that $x,$ $y,$ and $z$ are three positive numbers that satisfy the equations $xyz = 1,$ $x + \\frac {1}{z} = 5,$ and $y + \\frac {1}{x} = 29.$ Then $z + \\frac {1}{y} = \\frac {m}{n},$ where $m$ and $n$ are [[relatively prime]] positive integers. Find $m + n$. Show that it is 5.\n\n\nnote: this is the type of problem that makes you think symmetry, but actually can be solved easily with substitution, and other normal technniques\n\n### Solution\n\nWe can rewrite $xyz=1$ as $\\frac{1}{z}=xy$.\n\nSubstituting into one of the given equations, we have \n$x+xy=5$\n$x(1+y)=5$\n$\\frac{1}{x}=\\frac{1+y}{5}.$\n\nWe can substitute back into $y+\\frac{1}{x}=29$ to obtain\n$y+\\frac{1+y}{5}=29$\n$5y+1+y=145$\n$y=24.$\n\nWe can then substitute once again to get\n$x=\\frac15$\n$z=\\frac{5}{24}.$\nThus, $z+\\frac1y=\\frac{5}{24}+\\frac{1}{24}=\\frac{1}{4}$, so $m+n=005$.*)\n\nFormal:\ntheorem\n  fixes x y z :: real\n    and p :: rat\n  assumes \"0 < x \\<and> 0 < y \\<and> 0 < z\"\n    and \"x * y * z = 1\"\n    and \"x + 1 / z = 5\"\n    and \"y + 1 / x = 29\"\n    and \"z + 1 / y = p\"\n    and \"0 < p\" \n  shows \"let (m,n) = quotient_of p in m + n = 5\"\nproof -\n  (* We can rewrite $xyz=1$ as $\\frac{1}{z}=xy$. *)\n  have c0: \"z = 1 / (x*y)\"\n    sledgehammer\n  (* Substituting into one of the given equations, we have \n  $x+xy=5$\n  $x(1+y)=5$\n  $\\frac{1}{x}=\\frac{1+y}{5}.$ *)\n  have c1: \"1 / x = (1+y) / 5\" \n  proof -\n    have \"x + x * y = 5\" using assms(3) unfolding c0\n      sledgehammer\n    then have \"x * (1 + y) = 5\"\n      sledgehammer\n    then have t1: \"x = 5 / (1+y)\"\n      sledgehammer\n    then show ?thesis\n      sledgehammer\n  qed\n  (* We can substitute back into $y+\\frac{1}{x}=29$ to obtain\n  $y+\\frac{1+y}{5}=29$\n  $5y+1+y=145$\n  $y=24.$ *)\n  have \"y + (1+y)/5 = 29\" using assms(4) unfolding c1 sledgehammer\n  then have \"5* (y + (1+y)/5) = 5 * 29\" sledgehammer\n  also have \"... = 145\" sledgehammer\n  finally have c2_1: \"5* (y + (1+y)/5) = 145\" sledgehammer\n  have \"5* (y + (1+y)/5) = 5*y + (1+y)\" sledgehammer\n  also have \"... = 6*y + 1\" sledgehammer\n  finally have c2_2: \"5* (y + (1+y)/5) = 6*y + 1\" sledgehammer\n  have \"6*y + 1 = 145\" using c2_1 c2_2 sledgehammer\n  then have c2: \"y = 24\" sledgehammer\n  (* We can then substitute once again to get\n  $x=\\frac15$\n  $z=\\frac{5}{24}.$ *)\n  have \"1/x = 5\" using c1 unfolding c2 sledgehammer\n  then have c3: \"x = 1/5\"\n    sledgehammer\n  then have c4: \"z = 5/24\"\n    sledgehammer\n  (* Thus, $z+\\frac1y=\\frac{5}{24}+\\frac{1}{24}=\\frac{1}{4}$, so $m+n=005$. *)\n  have \"p = z + 1/y\" using assms(5) sledgehammer\n  also have \"... = 5/24 + 1/24\" unfolding c2 c4 sledgehammer\n  also have \"... = 1/4\" sledgehammer\n  finally have c5: \"p = 1/4\"\n    sledgehammer\n  have \"quotient_of p = (1, 4)\" unfolding c5 sledgehammer\n  then show ?thesis sledgehammer\nqed"},
    {"tag": "algebra_2rootsintpoly_am10tap11eqasqpam110", "category": "algebra", "metadata": {}, "prompt": "Informal:\n(*### Problem\n\nShow that for any complex number a, $(a-10)(a+11) = a^2 + a - 110$.\n\n### Solution\n\nWe first expand all terms of the left hand side to get $a^2 - 10a + 11a - 10*11$.\nThis equals $a^2 + a - 10*11 = a^2 + a - 110$.*)\n\nFormal:\ntheorem\n  fixes a :: complex\n  shows \"(a-10) * (a+11) = a^2 + a -110\"\nproof -\n  (* We first expand all terms of the left hand side to get $a^2 - 10a + 11a - 10*11$. *)\n  have \"(a-10) * (a+11) = a^2 - 10*a + 11*a - 10 *11\"\n    sledgehammer\n  (* This equals $a^2 + a - 10*11 = a^2 + a - 110$. *)\n  also have \"\\<dots> = a^2 + a - 10 * 11\"\n    sledgehammer\n  also have \"\\<dots> = a^2 + a - 110\"\n    sledgehammer\n  finally show ?thesis\n    sledgehammer\nqed"},
    {"tag": "mathd_numbertheory_335", "category": "number_theory", "metadata": {}, "prompt": "Informal:\n(*### Problem\n\nWhen Rachel divides her favorite number by 7, she gets a remainder of 5. What will the remainder be if she multiplies her favorite number by 5 and then divides by 7? Show that it is 4.\n\n### Solution\n\nLet $n$ be Rachel's favorite number. \nThen $n \\equiv 5 \\pmod{7}$, so $5n \\equiv 5 \\cdot 5 \\equiv 25 \\equiv 4 \\pmod{7}$.\n*)\n\nFormal:\ntheorem\n  fixes n :: nat\n  assumes h0 : \"n mod 7 = 5\"\n  shows \"(5 * n) mod 7 = 4\"\nproof -\n  (* Then $n \\equiv 5 \\pmod{7}$, so $5n \\equiv 5 \\cdot 5 \\equiv 25 \\equiv 4 \\pmod{7}$. *)\n  have c0:\"(5 * n) mod 7 = (5 * 5) mod 7\" using h0\n    sledgehammer\n  then have \"\\<dots> = 4\" sledgehammer\n  then have \"(5 * n) mod 7 = 4\" using c0 sledgehammer\n  then show ?thesis sledgehammer\nqed"}
]

examples_for_dsp_draft_prompt_template = [
    {
        "tag": "aimeI_2000_p7",
        "category": "algebra",
        "metadata": {},
        "prompt": (
            "Informal:\n"
            "(*### Problem\n\n"
            "Suppose that $x,$ $y,$ and $z$ are three positive numbers that satisfy the equations $xyz = 1,$ "
            "$x + \\frac {1}{z} = 5,$ and $y + \\frac {1}{x} = 29.$ Then $z + \\frac {1}{y} = \\frac {m}{n},$ "
            "where $m$ and $n$ are [[relatively prime]] positive integers. Find $m + n$. Show that it is 5.\n\n"
            "note: this is the type of problem that makes you think symmetry, but actually can be solved easily "
            "with substitution, and other normal technniques\n\n"
            "### Solution\n\n"
            "We can rewrite $xyz=1$ as $\\frac{1}{z}=xy$.\n\n"
            "Substituting into one of the given equations, we have \n$x+xy=5$\n$x(1+y)=5$\n$\\frac{1}{x}=\\frac{1+y}{5}.$\n\n"
            "We can substitute back into $y+\\frac{1}{x}=29$ to obtain\n"
            "$y+\\frac{1+y}{5}=29$\n$5y+1+y=145$\n$y=24.$\n\n"
            "We can then substitute once again to get\n$x=\\frac15$\n$z=\\frac{5}{24}.$\n"
            "Thus, $z+\\frac1y=\\frac{5}{24}+\\frac{1}{24}=\\frac{1}{4}$, so $m+n=005$.*)\n\n"
            "Formal:\n"
            "theorem\n"
            "  fixes x y z :: real\n"
            "    and p :: rat\n"
            "  assumes \"0 < x \\<and> 0 < y \\<and> 0 < z\"\n"
            "    and \"x * y * z = 1\"\n"
            "    and \"x + 1 / z = 5\"\n"
            "    and \"y + 1 / x = 29\"\n"
            "    and \"z + 1 / y = p\"\n"
            "    and \"0 < p\" \n"
            "  shows \"let (m,n) = quotient_of p in m + n = 5\"\n"
            "proof -\n"
            "  (* We can rewrite $xyz=1$ as $\\frac{1}{z}=xy$. *)\n"
            "  have c0: \"z = 1 / (x*y)\"\n"
            "    sledgehammer\n"
            "  (* Substituting into one of the given equations, we have \n"
            "  $x+xy=5$\n"
            "  $x(1+y)=5$\n"
            "  $\\frac{1}{x}=\\frac{1+y}{5}.$ *)\n"
            "  have c1: \"1 / x = (1+y) / 5\" \n"
            "  proof -\n"
            "    have \"x + x * y = 5\" using assms(3) unfolding c0\n"
            "      sledgehammer\n"
            "    then have \"x * (1 + y) = 5\"\n"
            "      sledgehammer\n"
            "    then have t1: \"x = 5 / (1+y)\"\n"
            "      sledgehammer\n"
            "    then show ?thesis\n"
            "      sledgehammer\n"
            "  qed\n"
            "  (* We can substitute back into $y+\\frac{1}{x}=29$ to obtain\n"
            "  $y+\\frac{1+y}{5}=29$\n"
            "  $5y+1+y=145$\n"
            "  $y=24.$ *)\n"
            "  have \"y + (1+y)/5 = 29\" using assms(4) unfolding c1 sledgehammer\n"
            "  then have \"5* (y + (1+y)/5) = 5 * 29\" sledgehammer\n"
            "  also have \"... = 145\" sledgehammer\n"
            "  finally have c2_1: \"5* (y + (1+y)/5) = 145\" sledgehammer\n"
            "  have \"5* (y + (1+y)/5) = 5*y + (1+y)\" sledgehammer\n"
            "  also have \"... = 6*y + 1\" sledgehammer\n"
            "  finally have c2_2: \"5* (y + (1+y)/5) = 6*y + 1\" sledgehammer\n"
            "  have \"6*y + 1 = 145\" using c2_1 c2_2 sledgehammer\n"
            "  then have c2: \"y = 24\" sledgehammer\n"
            "  (* We can then substitute once again to get\n"
            "  $x=\\frac15$\n"
            "  $z=\\frac{5}{24}.$ *)\n"
            "  have \"1/x = 5\" using c1 unfolding c2 sledgehammer\n"
            "  then have c3: \"x = 1/5\"\n"
            "    sledgehammer\n"
            "  then have c4: \"z = 5/24\"\n"
            "    sledgehammer\n"
            "  (* Thus, $z+\\frac1y=\\frac{5}{24}+\\frac{1}{24}=\\frac{1}{4}$, so $m+n=005$. *)\n"
            "  have \"p = z + 1/y\" using assms(5) sledgehammer\n"
            "  also have \"... = 5/24 + 1/24\" unfolding c2 c4 sledgehammer\n"
            "  also have \"... = 1/4\" sledgehammer\n"
            "  finally have c5: \"p = 1/4\"\n"
            "    sledgehammer\n"
            "  have \"quotient_of p = (1, 4)\" unfolding c5 sledgehammer\n"
            "  then show ?thesis sledgehammer\n"
            "qed"
        ),
    },
    {
        "tag": "algebra_2rootsintpoly_am10tap11eqasqpam110",
        "category": "algebra",
        "metadata": {},
        "prompt": (
            "Informal:\n"
            "(*### Problem\n\n"
            "Show that for any complex number a, $(a-10)(a+11) = a^2 + a - 110$.\n\n"
            "### Solution\n\n"
            "We first expand all terms of the left hand side to get $a^2 - 10a + 11a - 10*11$.\n"
            "This equals $a^2 + a - 10*11 = a^2 + a - 110$.*)\n\n"
            "Formal:\n"
            "theorem\n"
            "  fixes a :: complex\n"
            "  shows \"(a-10) * (a+11) = a^2 + a -110\"\n"
            "proof -\n"
            "  (* We first expand all terms of the left hand side to get $a^2 - 10a + 11a - 10*11$. *)\n"
            "  have \"(a-10) * (a+11) = a^2 - 10*a + 11*a - 10 *11\"\n"
            "    sledgehammer\n"
            "  (* This equals $a^2 + a - 10*11 = a^2 + a - 110$. *)\n"
            "  also have \"\\<dots> = a^2 + a - 10 * 11\"\n"
            "    sledgehammer\n"
            "  also have \"\\<dots> = a^2 + a - 110\"\n"
            "    sledgehammer\n"
            "  finally show ?thesis\n"
            "    sledgehammer\n"
            "qed"
        ),
    },
    {
        "tag": "mathd_numbertheory_335",
        "category": "number_theory",
        "metadata": {},
        "prompt": (
            "Informal:\n"
            "(*### Problem\n\n"
            "When Rachel divides her favorite number by 7, she gets a remainder of 5. What will the remainder be if she "
            "multiplies her favorite number by 5 and then divides by 7? Show that it is 4.\n\n"
            "### Solution\n\n"
            "Let $n$ be Rachel's favorite number. \n"
            "Then $n \\equiv 5 \\pmod{7}$, so $5n \\equiv 5 \\cdot 5 \\equiv 25 \\equiv 4 \\pmod{7}$.\n*)\n\n"
            "Formal:\n"
            "theorem\n"
            "  fixes n :: nat\n"
            "  assumes h0 : \"n mod 7 = 5\"\n"
            "  shows \"(5 * n) mod 7 = 4\"\n"
            "proof -\n"
            "  (* Then $n \\equiv 5 \\pmod{7}$, so $5n \\equiv 5 \\cdot 5 \\equiv 25 \\equiv 4 \\pmod{7}$. *)\n"
            "  have c0:\"(5 * n) mod 7 = (5 * 5) mod 7\" using h0\n"
            "    sledgehammer\n"
            "  then have \"\\<dots> = 4\" sledgehammer\n"
            "  then have \"(5 * n) mod 7 = 4\" using c0 sledgehammer\n"
            "  then show ?thesis sledgehammer\n"
            "qed"
        ),
    }
]

# -- Prompts for generating (informal) drafts (basically informal/natural language solution strings, that contain less details than a formal proof, hence why they are called "drafts")
prompt_draft_template_4_isabelle = """Draft an informal solution similar to the one below. 
The informal solution will be used to sketch a formal Isabelle proof.
Here are some examples: \n"""
for example in examples_for_dsp_draft_prompt_template:
    # P_draft_isa_prompt_template += ("Example:\n" + x['prompt'][:x['prompt'].find('Formal:')] + "\n\n")
    # - Extract the 'prompt' field from the current example
    prompt_text = example['prompt']
    # - Find the index where the 'Formal:' keyword starts
    formal_index = prompt_text.find('Formal:')
    # - Extract the part of the prompt before the 'Formal:' keyword
    nl_prob_soln = prompt_text[:formal_index]  # Append nl/i draft examples: prob_nl, soln_nl/draft_nl
    # - Append this i draft example our prompt draft/P_draft
    prompt_draft_template_4_isabelle += f"Example:\n{informal_part}\n\n"
# append the final part of the prompt template that prompts model to do prediction, we'd need to insert new nl problem text here before using it
prompt_draft_template_4_isabelle += """Informal:
(*### Problem

"""

# P_sketch isabelle, ref: https://github.com/brando90/ntptutorial-II/blob/main/partII_dsp/notebooks/II_dsp__part2_dsp.ipynb
prompt = """Translate the informal solution into a sketch of the
formal Isabelle proof. Add `sledgehammer` in the sketch whenever
possible. `sledgehammer` will be used to call the automated Sledgehammer prover. 
Here are some examples:
"""
for x in examples:
    prompt += (x['prompt'] + "\n\n")
prompt += """Informal:
(*### Problem

"""

xf = """theorem
fixes x :: int
assumes h0: "even x"
shows "odd (x+5)" """

zi = p.f(prompt, xi + '\n\n' + yi + '\n\n' + xf)
print(zi)