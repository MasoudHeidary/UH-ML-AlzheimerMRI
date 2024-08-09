import time

class Log:

    def __init__(self, name="log.txt", terminal=True) -> None:
        self.f = open(name, "a")
        self.terminal = terminal

    def __lshift__(self, txt):
        txt = f"[{time.asctime()}] >> " + txt
        if self.terminal:
            print(txt, end='')
        self.f.write(txt)
        self.f.flush()

    def print(self, txt=""):
        self << txt

    def println(self, txt=""):
        self << (txt + "\n")

    def __del__(self):
        self.f.flush()
        self.f.close()
