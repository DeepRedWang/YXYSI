import datetime
import time

class Timer():
    def __init__(self):
        self.contain = []
        print(f"Creat Timer at {datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")

    def start(self):
        self.tik = time.time()

    def end(self):
        self.contain.append(time.time() - self.tik)
        return self.contain[-1]

    def avg(self):
        return sum(self.contain) / len(self.contain)
