from ._eddie import eddie_profile


def get_profile(profile: str) -> dict:
    assert isinstance(profile, str), "Profile must be a string."
    if profile == "eddie":
        return eddie_profile.copy()
    else:
        raise ValueError(f"Profile {profile} not found.")
