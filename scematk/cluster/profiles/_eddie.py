"""
Profile for connecting to Eddie, the University of Edinburgh's HPC cluster.

Profile contact: Hugh Warden - h.b.warden@sms.ed.ac.uk
"""

eddie_profile = {
    "cluster_type": "SGE",
    "memory": "8 G",
    "cores": 1,
    "resource_spec": "h_vmem=8G",
    "worker_extra_args": ["--lifetime", "25m", "--lifetime-stagger", "4m"],
}
