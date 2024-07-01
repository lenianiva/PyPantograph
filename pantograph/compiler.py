from dataclasses import dataclass

@dataclass(frozen=True)
class TacticInvocation:
    before: str
    after: str
    tactic: str

    @staticmethod
    def parse(payload: dict):
        return TacticInvocation(before=payload["goalBefore"],
                                after=payload["goalAfter"],
                                tactic=payload["tactic"])
