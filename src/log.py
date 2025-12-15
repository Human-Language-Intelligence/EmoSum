class Logger:
    enabled = False

    def __init__(self, enabled=False):
        print(f"Init logger with {enabled}")
        self.enabled = enabled

    def log(self, msg):
        if self.enabled:
            print(msg)

    def force_log(self, msg):
        print(msg)
