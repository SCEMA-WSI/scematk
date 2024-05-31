from ._eddie import eddie_profile


def get_profile(profile: str) -> dict:
    """Get a profile for a specific cluster.

    Args:
        profile (str): The name of the profile to get.

    Raises:
        ValueError: If the profile is not found.

    Returns:
        dict: The profile.
    """
    assert isinstance(profile, str), "Profile must be a string."
    if profile == "eddie":
        return eddie_profile.copy()
    else:
        raise ValueError(f"Profile {profile} not found.")
