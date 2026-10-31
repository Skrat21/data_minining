# compatibility shim for huggingface_hub BucketNotFoundError
# and other removed names in newer huggingface_hub versions
# transformers 4.39.0+ imports datasets which requires BucketNotFoundError
# from huggingface_hub.errors, but newer huggingface_hub (>=0.30) removed it.
# This shim maps missing names to their alternatives for compatibility.

import sys

# Apply patches to huggingface_hub modules BEFORE they are used by datasets/transformers
# This must happen early, before any other huggingface_hub submodules are imported

# Patch hf_file_system to add missing names
try:
    import huggingface_hub.hf_file_system as hf_fs
    if not hasattr(hf_fs, 'HfFileSystemResolvedBucketPath'):
        hf_fs.HfFileSystemResolvedBucketPath = hf_fs.HfFileSystemResolvedPath
    if not hasattr(hf_fs, 'HfFileSystemResolvedRepositoryPath'):
        hf_fs.HfFileSystemResolvedRepositoryPath = hf_fs.HfFileSystemResolvedPath
except ImportError:
    pass

# Patch huggingface_hub.errors to add missing names
try:
    import huggingface_hub.errors as hf_errors
    # BucketNotFoundError -> EntryNotFoundError
    if not hasattr(hf_errors, 'BucketNotFoundError'):
        from huggingface_hub.errors import EntryNotFoundError
        hf_errors.BucketNotFoundError = EntryNotFoundError
except ImportError:
    pass

# Patch huggingface_hub.utils to add missing httpx name
try:
    import huggingface_hub.utils as hf_utils
    if not hasattr(hf_utils, 'httpx'):
        # Create a minimal httpx mock for compatibility
        class _HttpxMock:
            class RequestError(Exception):
                pass
            class Timeout(Exception):
                pass
            class ConnectError(Exception):
                pass
            class TooManyRedirects(Exception):
                pass
            class HTTPStatusError(Exception):
                pass
            class Request:
                def __init__(self, *args, **kwargs):
                    pass
                def send(self, *args, **kwargs):
                    import requests
                    return requests.request(*args, **kwargs)
                def raise_for_status(self):
                    pass
            Request = Request
            def get(*args, **kwargs):
                import requests
                return requests.get(*args, **kwargs)
            def post(*args, **kwargs):
                import requests
                return requests.post(*args, **kwargs)
        hf_utils.httpx = _HttpxMock()
except ImportError:
    pass

print("hf_compat: Compatibility shim applied for huggingface_hub")