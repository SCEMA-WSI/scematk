from dask.distributed import Client
from dask_jobqueue import SGECluster
from .profiles import eddie_profile

def quick_connect(profile: str) -> Client:
    assert isinstance(profile, str), "profile must be a string"
    if profile == "eddie":
        cluster = SGECluster(**eddie_profile)
        client = Client(cluster)
        return client
    else:
        raise ValueError(f"Profile {profile} not recognised")