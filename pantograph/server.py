import json, pexpect, pathlib

def _get_proc_path():
    return pathlib.Path(__file__).parent / "pantograph"

class Server:

    def __init__(self,
                 imports=["Init"],
                 options=[],
                 timeout=20,
                 maxread=1000000):
        """
        timeout: Amount of time to wait for execution
        maxread: Maximum number of characters to read (especially important for large proofs and catalogs)
        """
        self.timeout = timeout
        self.imports = imports
        self.maxread = maxread
        self.proc_path = _get_proc_path()

        self.options = options
        self.args = " ".join(imports + [f'--{opt}' for opt in options])
        self.proc = None
        self.restart()

    def restart(self):
        if self.proc is not None:
            self.proc.close()
        self.proc = pexpect.spawn(
            f"{self.proc_path} {self.args}",
            encoding="utf-8",
            maxread=self.maxread
        )
        self.proc.setecho(False)

    def run(self, cmd, payload):
        s = json.dumps(payload)
        self.proc.sendline(f"{cmd} {s}")
        try:
            self.proc.expect("{.*}\r\n", timeout=self.timeout)
            output = self.proc.match.group()
            return json.loads(output)
        except pexpect.exceptions.TIMEOUT:
            raise pexpect.exceptions.TIMEOUT

    def reset(self):
        return self.run("reset", {})

    def goal_start(self, expr: str):
        return self.run('goal.start', {"expr": str(expr)})
    def goal_tactic(self, stateId: int, goalId: int, tactic):
        return self.run('goal.tactic', {"stateId": stateId, "goalId": goalId, "tactic": tactic})


def check_version():
    import subprocess
    with subprocess.Popen([_get_proc_path(), "--version"], stdout=subprocess.PIPE) as p:
        v = p.communicate()[0].decode('utf-8').strip()
        print(v)


if __name__ == '__main__':
    check_version()
