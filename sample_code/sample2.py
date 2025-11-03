class APIHandler:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_data(self, endpoint):
        """Fetches data from an API endpoint."""
        return f"Fetching from {endpoint} with key {self.api_key}"

def utility_function_alpha():
    """A utility function."""
    return "Utility Alpha"
