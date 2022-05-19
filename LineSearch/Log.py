class Log:
    def __init__(self):
        self.history = None
            
    def log(self, **kwargs):
        if self.history is None:
            self.history = {key: [val] for key, val in kwargs.items()}
            return
            
        for key, val in kwargs.items():
            log_val = self.history.get(key, None)
            
            if log_val is None:
                raise ValueError("Logging callback must have the same keywords from the beginning of the log."
                                 f" But got a new kwarg {key}")
            
            log_val.append(val)
