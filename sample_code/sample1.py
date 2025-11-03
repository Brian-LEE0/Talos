class DataProcessor:
    def __init__(self, name):
        self.name = name

    def process(self, data):
        """Processes the given data."""
        return f"Processed {data} with {self.name}"

def helper_function_one():
    """A helper function."""
    return "Helper One"
