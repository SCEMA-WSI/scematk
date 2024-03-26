from dask_jobqueue import SGECluster
from .profiles import eddie_profile

def quick_connect(profile: str, min: int = 14, max: int = 16):
    assert isinstance(profile, str), "profile must be a string"
    if profile == "eddie":
        cluster = SGECluster(**eddie_profile)
    else:
        raise ValueError(f"Profile {profile} not recognised")
    cluster.adapt(minimum=min, maximum=max)
    cluster.wait_for_workers(min)
    return cluster