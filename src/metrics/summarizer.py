class Summarizer:
    def __init__(self, metrics: list):
        self.metrics = metrics
    
    def __call__(self, *args, **kwargs):
        result = {}
        for metric in self.metrics:
            result[metric.name] = metric(*args, **kwargs)

        return result
